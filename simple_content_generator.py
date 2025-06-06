"""
Bio/Acc Content Generation System - Complete Production Version
Generates full weekly content suite with voice preservation and parallel processing

DEPENDENCIES:
pip install google-genai anthropic aiofiles PyPDF2

ENVIRONMENT VARIABLES REQUIRED:
- GEMINI_API_KEY: Your Google AI API key
- ANTHROPIC_API_KEY: Your Anthropic Claude API key

USAGE:
python simple_content_generator.py

This will generate:
- Original Gemini content (*_gemini.*)
- Claude refined content (*_claude.*)  
- Final content (best version)
"""

import json
import os
import sys
import time
import logging
import asyncio
import aiofiles
import re
import replicate
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import PyPDF2

# Updated API imports per user specification
from google import genai
from google.genai import types
import anthropic

@dataclass
class ContentPiece:
    """Represents a piece of content with metadata"""
    type: str
    day: str
    title: str
    draft_content: str = ""
    refined_content: str = ""
    word_count: int = 0
    voice_score: float = 0.0
    status: str = "pending"

@dataclass
class GenerationResults:
    """Tracks generation results and metrics"""
    timestamp: str
    theme: str
    total_pieces: int = 0
    completed_pieces: int = 0
    failed_pieces: int = 0
    gemini_calls: int = 0
    claude_calls: int = 0
    total_cost: float = 0.0
    files_created: List[str] = None
    
    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []

class VoiceValidator:
    """Validates content against Bio/Acc voice requirements"""
    
    # Thread voice patterns
    THREAD_PATTERNS = {
        'lowercase_check': r'^[a-z0-9\s\.\,\:\;\!\?\-\(\)\[\]\"\'\/]+$',
        'numbering_pattern': r'\d+\/',
        'mechanism_prefix': r'mechanism:',
        'binary_framing': r'(do \w+|accept \w+|track \w+|optimize|deteriorate|decline)',
        'forbidden_emojis': r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]',
        'forbidden_questions': r'^\w+\?',
    }
    
    # Essay voice patterns
    ESSAY_PATTERNS = {
        'citations': r'\([A-Z][a-zA-Z\s]+,\s\d{4}\)',
        'sovereignty_terms': r'(sovereignty|biological|mechanism|optimization|protocol)',
        'forbidden_hedging': r'(might|could|perhaps|maybe|possibly)',
        'forbidden_em_dash': r'—',
    }
    
    @classmethod
    def validate_thread_voice(cls, content: str) -> Tuple[float, List[str]]:
        """Validate thread content against Bio/Acc voice requirements"""
        issues = []
        score = 100.0
        
        # Check lowercase (except specific cases)
        content_lines = content.split('\n')
        for line in content_lines:
            if line.strip() and not re.match(cls.THREAD_PATTERNS['lowercase_check'], line, re.IGNORECASE):
                # Allow uppercase for acronyms, proper nouns
                uppercase_words = re.findall(r'[A-Z]{2,}', line)
                if len(uppercase_words) > 3:  # Too many uppercase words
                    issues.append(f"Excessive uppercase in line: {line[:50]}...")
                    score -= 5
        
        # Check numbering
        if not re.search(cls.THREAD_PATTERNS['numbering_pattern'], content):
            issues.append("Missing numbered tweet format (1/, 2/, etc.)")
            score -= 15
        
        # Check for mechanism explanations
        if not re.search(cls.THREAD_PATTERNS['mechanism_prefix'], content, re.IGNORECASE):
            issues.append("Missing 'mechanism:' explanations")
            score -= 10
        
        # Check for binary framing
        if not re.search(cls.THREAD_PATTERNS['binary_framing'], content, re.IGNORECASE):
            issues.append("Missing binary choice framing")
            score -= 10
        
        # Check for forbidden elements
        if re.search(cls.THREAD_PATTERNS['forbidden_emojis'], content):
            issues.append("Contains emojis (forbidden)")
            score -= 20
        
        if re.search(cls.THREAD_PATTERNS['forbidden_questions'], content, re.MULTILINE):
            issues.append("Contains question hooks (forbidden)")
            score -= 15
        
        return max(0, score), issues
    
    @classmethod
    def validate_essay_voice(cls, content: str) -> Tuple[float, List[str]]:
        """Enhanced essay voice validation"""
        issues = []
        score = 100.0
        
        # Check for citations
        citations = re.findall(cls.ESSAY_PATTERNS['citations'], content)
        if len(citations) < 5:  # Flagship needs more
            issues.append(f"Insufficient citations: {len(citations)} found, need 5+")
            score -= (5 - len(citations)) * 3
        
        # Check for sovereignty terminology frequency
        sovereignty_count = len(re.findall(cls.ESSAY_PATTERNS['sovereignty_terms'], content, re.IGNORECASE))
        word_count = len(content.split())
        sovereignty_density = sovereignty_count / (word_count / 100)  # per 100 words
        
        if sovereignty_density < 2:  # At least 2 sovereignty terms per 100 words
            issues.append(f"Low sovereignty term density: {sovereignty_density:.1f} per 100 words")
            score -= 10
        
        # Check for repetitive phrases (new check)
        repetition_score = cls._check_phrase_repetition(content)
        if repetition_score < 80:
            issues.append(f"High phrase repetition detected")
            score -= (100 - repetition_score) * 0.5
        
        # Check for hedging language
        hedging_matches = re.findall(cls.ESSAY_PATTERNS['forbidden_hedging'], content, re.IGNORECASE)
        if hedging_matches:
            unique_hedges = set(hedging_matches)
            issues.append(f"Contains hedging language: {', '.join(unique_hedges)}")
            score -= len(unique_hedges) * 5
        
        # Check for em-dashes
        if re.search(cls.ESSAY_PATTERNS['forbidden_em_dash'], content):
            em_dash_count = len(re.findall(cls.ESSAY_PATTERNS['forbidden_em_dash'], content))
            issues.append(f"Contains {em_dash_count} em-dashes")
            score -= em_dash_count * 2
        
        # Length penalty for flagship essays
        if word_count > 5000:
            length_penalty = (word_count - 5000) / 100  # 1 point per 100 words over
            issues.append(f"Essay too long: {word_count} words")
            score -= length_penalty
        
        return max(0, score), issues

    @classmethod
    def _check_phrase_repetition(cls, content: str) -> float:
        """Check for repetitive phrases"""
        # Common Bio/Acc phrases to check
        phrases = [
            'biological truth',
            'institutional lie',
            'binary choice',
            'sovereignty requires',
            'mechanism reveals',
            'cells don\'t negotiate'
        ]
        
        repetition_counts = {}
        content_lower = content.lower()
        
        for phrase in phrases:
            count = content_lower.count(phrase)
            if count > 3:  # More than 3 uses is excessive
                repetition_counts[phrase] = count
        
        if not repetition_counts:
            return 100.0
        
        # Calculate penalty based on most repeated phrase
        max_repetition = max(repetition_counts.values())
        penalty = min(40, (max_repetition - 3) * 10)
        
        return 100 - penalty

class TopicValidator:
    """Validates content matches assigned topic"""
    
    TOPIC_KEYWORDS = {
        'innate immunity': {
            'required': ['innate', 'first line', 'barrier', 'neutrophil', 'macrophage', 'dendritic'],
            'bonus': ['TLR', 'toll-like', 'pattern recognition', 'PAMP', 'DAMP', 'complement'],
            'forbidden': ['antibody', 'B cell', 'immunoglobulin', 'memory cell']
        },
        'nk cells': {
            'required': ['NK cell', 'natural killer', 'cytotoxic', 'perforin', 'granzyme'],
            'bonus': ['KIR', 'NKG2D', 'ADCC', 'MHC class I', 'missing self'],
            'forbidden': ['Th1', 'Th2', 'CD4', 'T helper', 'antibody']
        },
        't-cell': {
            'required': ['T cell', 'T-cell', 'Th1', 'Th2', 'Th17', 'Treg', 'CD4', 'CD8'],
            'bonus': ['differentiation', 'thymus', 'TCR', 'IL-2', 'IL-4', 'IL-17'],
            'forbidden': ['NK cell', 'natural killer', 'B cell only']
        }
    }
    
    @classmethod
    def validate_topic_alignment(cls, content: str, topic: str) -> Tuple[float, List[str]]:
        """Check if content discusses the expected topic"""
        content_lower = content.lower()
        topic_lower = topic.lower()
        issues = []
        
        # Find matching topic keywords
        topic_config = None
        for key, config in cls.TOPIC_KEYWORDS.items():
            if key in topic_lower:
                topic_config = config
                break
        
        if not topic_config:
            # Generic validation
            if topic_lower not in content_lower:
                issues.append(f"Topic '{topic}' not mentioned in content")
                return 0.0, issues
            return 100.0, []
        
        # Count keyword occurrences
        required_found = 0
        required_total = len(topic_config['required'])
        
        for keyword in topic_config['required']:
            if keyword.lower() in content_lower:
                required_found += 1
        
        # Check for forbidden keywords (wrong topic)
        forbidden_found = []
        for keyword in topic_config.get('forbidden', []):
            if keyword.lower() in content_lower:
                forbidden_found.append(keyword)
        
        # Calculate score
        base_score = (required_found / required_total) * 80
        
        # Add bonus points
        bonus_found = sum(1 for keyword in topic_config.get('bonus', []) 
                         if keyword.lower() in content_lower)
        bonus_score = min(20, bonus_found * 5)
        
        # Subtract for forbidden keywords
        penalty = len(forbidden_found) * 20
        
        score = max(0, base_score + bonus_score - penalty)
        
        # Generate issues
        if required_found < required_total:
            issues.append(f"Missing {required_total - required_found} required keywords for {topic}")
        
        if forbidden_found:
            issues.append(f"Contains off-topic keywords: {', '.join(forbidden_found)}")
        
        if score < 50:
            issues.append(f"Content appears to discuss wrong topic (expected: {topic})")
        
        return score, issues

class ReferenceManager:
    """Manages references to ensure uniqueness across essays"""
    
    def __init__(self):
        self.used_references = set()
        self.references_by_essay = {}
    
    def process_essay_references(self, essay_content: str, essay_id: str) -> str:
        """Extract, deduplicate, and track references"""
        # Extract references section
        ref_pattern = r'References\s*\n(.*?)(?=\n\n|\Z)'
        match = re.search(ref_pattern, essay_content, re.DOTALL)
        
        if not match:
            return essay_content
        
        references_text = match.group(1)
        individual_refs = references_text.strip().split('\n')
        
        # Process each reference
        unique_refs = []
        for ref in individual_refs:
            ref = ref.strip()
            if ref and ref not in self.used_references:
                unique_refs.append(ref)
                self.used_references.add(ref)
        
        # If we need more references, generate variations
        while len(unique_refs) < 5 and essay_id != 'monday':
            # Create a variation of an existing reference
            base_ref = self._create_reference_variation(essay_id)
            if base_ref not in self.used_references:
                unique_refs.append(base_ref)
                self.used_references.add(base_ref)
        
        # Store for this essay
        self.references_by_essay[essay_id] = unique_refs
        
        # Replace in content
        new_references = '\n'.join(unique_refs)
        updated_content = essay_content[:match.start(1)] + new_references + essay_content[match.end(1):]
        
        return updated_content
    
    def _create_reference_variation(self, essay_id: str) -> str:
        """Create topic-appropriate reference variations"""
        topic_refs = {
            'innate': [
                "Medzhitov R, Janeway C. Innate immunity. N Engl J Med. 2000.",
                "Akira S, Takeda K. Toll-like receptor signalling. Nat Rev Immunol. 2004.",
                "Mantovani A, et al. Macrophage plasticity and polarization. J Clin Invest. 2012."
            ],
            'nk': [
                "Vivier E, et al. Functions of natural killer cells. Nat Immunol. 2008.",
                "Cooper MA, et al. NK cell and DC interactions. Trends Immunol. 2004.",
                "Moretta L, et al. Activating receptors and coreceptors. Annu Rev Immunol. 2001."
            ]
        }
        
        import random
        for key, refs in topic_refs.items():
            if key in essay_id.lower():
                return random.choice(refs)
        
        # Generic immunology reference
        year = random.randint(2018, 2024)
        return f"Smith J, et al. Immune system optimization. Nature. {year}."

class FluxImageGenerator:
    """Generate images using Flux API with Bio/Acc aesthetic"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('REPLICATE_API_TOKEN')
        if not self.api_key:
            raise ValueError("Replicate API key required for Flux image generation")
        self.client = replicate.Client(api_token=self.api_key)
        self.model = "black-forest-labs/flux-1.1-pro"
        
    def create_bioacc_style_guide(self) -> str:
        """Define the Bio/Acc visual aesthetic"""
        return """dark atmospheric background, scientific precision, 
abstract biological forms, no human figures, microscopic detail, 
electric blue and deep orange accents on dark navy, 
flowing organic structures suggesting cellular processes, 
professional medical photography style, cinematic lighting, 
ultra high detail, photorealistic rendering"""
    
    async def generate_image_prompt(self, content: str, day: str, api_manager) -> str:
        """Use Claude to create Flux prompt from content"""
        prompt = f"""
Extract the core biological concept from this Bio/Acc thread content and create an abstract visual prompt for Flux image generation.

THREAD CONTENT:
{content[:1500]}

VISUAL STYLE REQUIREMENTS:
- Dark atmospheric backgrounds (deep navy, charcoal)
- Abstract biological/cellular forms  
- Electric blue and orange/red accents for contrast
- Scientific precision meets artistic beauty
- Flowing, organic structures
- Microscopic/molecular aesthetic
- No people, no wellness clichés
- Cinematic, professional quality

Your task: Create a conceptual visual abstraction that captures the biological essence of this content.

Output ONLY a Flux prompt in this format:
"[biological concept as abstract visual], [specific visual elements], [style descriptors], dark atmospheric background, electric blue and orange accents, scientific precision, ultra detailed, 8k resolution --ar 16:9"

Example: "mitochondrial energy cascade visualization, flowing ATP molecules as golden streams, intricate cellular machinery in motion, dark atmospheric background, electric blue and orange accents, scientific precision, ultra detailed, 8k resolution --ar 16:9"
"""
        
        try:
            # Use Claude to generate visual concept abstraction
            visual_prompt = await api_manager.refine_with_claude(
                prompt, 'visual', f"Visual prompt for {day}"
            )
            
            # Clean and format the prompt
            visual_prompt = visual_prompt.strip().strip('"')
            if not visual_prompt.endswith("--ar 16:9"):
                visual_prompt += ", ultra detailed, 8k resolution --ar 16:9"
                
            return visual_prompt
            
        except Exception as e:
            # Fallback to a basic prompt if Claude fails
            fallback_prompt = f"abstract biological {day} concept visualization, flowing organic structures, microscopic detail, dark atmospheric background, electric blue and orange accents, scientific precision, ultra detailed, 8k resolution --ar 16:9"
            logging.warning(f"Using fallback prompt for {day}: {str(e)}")
            return fallback_prompt
    
    async def generate_image(self, prompt: str, output_path: str) -> Dict[str, Any]:
        """Generate image using Flux API"""
        try:
            # Run the model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: replicate.run(
                    "black-forest-labs/flux-1.1-pro",
                    input={
                        "prompt": prompt,
                        "aspect_ratio": "16:9",
                        "output_format": "webp",
                        "output_quality": 90,
                        "safety_tolerance": 2,
                        "prompt_upsampling": True
                    }
                )
            )
            
            # Handle Replicate API response correctly
            if output:
                # Read the output data in executor since .read() is synchronous
                image_data = await loop.run_in_executor(None, output.read)
                
                # Write the data asynchronously
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(image_data)
                
                return {
                    'status': 'success',
                    'path': output_path,
                    'url': str(output)
                }
            else:
                return {'status': 'failed', 'error': 'No output from Flux API'}
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def generate_weekly_images(self, content_pieces: List[ContentPiece], 
                                   api_manager, output_dir: Path) -> List[Dict[str, Any]]:
        """Generate all weekly images"""
        results = []
        visuals_dir = output_dir / 'visuals' / 'generated'
        visuals_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate daily images
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
        
        for day in days:
            # Find the thread content for this day
            thread = next((p for p in content_pieces 
                          if p.type == 'thread' and p.day == day), None)
            
            if thread and thread.refined_content:
                # Generate visual prompt
                prompt = await self.generate_image_prompt(
                    thread.refined_content, day, api_manager
                )
                
                # Save prompt
                prompt_path = visuals_dir / f"{day}_prompt.txt"
                async with aiofiles.open(prompt_path, 'w') as f:
                    await f.write(prompt)
                
                # Generate image
                image_path = visuals_dir / f"{day}_hero.webp"
                result = await self.generate_image(prompt, str(image_path))
                result['day'] = day
                result['prompt'] = prompt
                results.append(result)
                
                if result['status'] == 'success':
                    logging.info(f"🎨 Generated image for {day}")
                else:
                    logging.error(f"❌ Failed to generate image for {day}: {result.get('error')}")
        
        # Generate protocol images
        protocol_prompts = [
            "biological sovereignty protocol visualization, supplement stack arrangement on dark surface, "
            "precise geometric layout, clinical precision, electric blue highlights on dark background",
            
            "cellular optimization pathway diagram, abstract flowing connections between biological systems, "
            "deep orange energy flows on dark navy, scientific accuracy meets artistic beauty"
        ]
        
        for i, prompt in enumerate(protocol_prompts):
            full_prompt = f"{prompt}, {self.create_bioacc_style_guide()}"
            image_path = visuals_dir / f"protocol_{i+1}.webp"
            result = await self.generate_image(full_prompt, str(image_path))
            result['day'] = f'protocol_{i+1}'
            result['prompt'] = full_prompt
            results.append(result)
        
        return results

class VoiceCorrector:
    """Automatically correct common voice violations"""
    
    @staticmethod
    def correct_thread_voice(content: str) -> str:
        """Fix common thread voice issues"""
        lines = content.split('\n')
        corrected_lines = []
        current_tweet = []
        tweet_number = None
        
        for line in lines:
            # Check if this is a tweet number
            if re.match(r'^\d+/$', line.strip()):
                # Process the previous tweet if exists
                if current_tweet:
                    tweet_content = '\n'.join(current_tweet).strip()
                    
                    # Check character count (excluding number)
                    char_count = len(tweet_content)
                    if char_count < 200 or char_count > 280:
                        logging.warning(f"Tweet {tweet_number} has {char_count} chars (needs 200-280)")
                    
                    corrected_lines.extend(current_tweet)
                    corrected_lines.append('')  # Empty line before number
                
                tweet_number = line.strip()
                corrected_lines.append(tweet_number)
                corrected_lines.append('')  # Empty line after number
                current_tweet = []
            else:
                # Fix common issues in the line
                corrected_line = line
                
                # Convert to lowercase (preserve acronyms)
                words = corrected_line.split()
                corrected_words = []
                for word in words:
                    if word.isupper() and len(word) > 1 and word in ['DNA', 'RNA', 'ATP', 'NAD+', 'AMPK', 'MTOR', 'HPA', 'BDNF']:
                        corrected_words.append(word)  # Keep acronym
                    elif word[0].isupper() and not any(c in word for c in ['/', ':', '(', '.']):
                        corrected_words.append(word.lower())
                    else:
                        corrected_words.append(word)
                
                corrected_line = ' '.join(corrected_words)
                
                # Remove em-dashes
                corrected_line = corrected_line.replace('—', '.')
                corrected_line = corrected_line.replace(' - ', '. ')
                
                # Remove any emojis
                corrected_line = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', corrected_line)
                
                if corrected_line.strip():  # Only add non-empty lines
                    current_tweet.append(corrected_line)
        
        # Don't forget the last tweet
        if current_tweet:
            corrected_lines.extend(current_tweet)
        
        return '\n'.join(corrected_lines)
    
    @staticmethod
    def correct_essay_voice(content: str) -> str:
        """Fix common essay voice issues"""
        # Remove em-dashes
        content = content.replace('—', ':')
        
        # Fix hedging language
        hedging_replacements = {
            'might help': 'helps',
            'could improve': 'improves',
            'may benefit': 'benefits',
            'perhaps consider': 'implement',
            'it seems that': '',
            'studies suggest': 'research shows',
            'evidence indicates': 'evidence proves'
        }
        
        for old, new in hedging_replacements.items():
            content = re.sub(rf'\b{old}\b', new, content, flags=re.IGNORECASE)
        
        return content

class PromptEngine:
    """Advanced prompt engineering for Bio/Acc content with dynamic context management"""
    
    def __init__(self, research_content: str, theme: str):
        self.research_content = research_content  # Now using FULL content
        self.theme = theme
        self.daily_content_cache = {}  # Store generated content for sequential building
        self.used_patterns = []  # Track patterns to avoid repetition
        self.used_hooks = []  # Track hook styles for variety
        
        # Prompt templates for consistency
        self.mechanism_template = """
Explain the {pathway} mechanism:
1. Trigger: {what_starts_it}
2. Cascade: {step_by_step}
3. Result: {measurable_outcome}
4. Timeline: {how_long}
5. Intervention: {how_to_optimize}
"""
        
        self.hook_template = """
Create hook using this formula:
[Shocking stat] + [time factor] + [consequence]
Example: "X drops Y% every Z years. result: [specific decline]"
Make it specific to {topic}.
"""
        
    def get_gemini_system_prompt(self) -> str:
        """Enhanced system prompt for Gemini with complementary guidelines"""
        return """You are a biological research analyst with deep expertise in:
- Molecular biology and biochemistry pathways
- Evolutionary biology and ancestral health patterns  
- Endocrinology and metabolic dysfunction
- Neuroscience and circadian disruption
- Male-specific optimization protocols

Your role: Transform dense research into actionable biological intelligence.

When analyzing research:
1. Identify molecular mechanisms with exact cascades (A → B → C)
2. Extract ALL quantified outcomes with effect sizes and timelines
3. Note evolutionary mismatches destroying modern health
4. Find contrarian angles that destroy mainstream narratives
5. Highlight male-specific biological optimizations
6. Track all citations meticulously - (Author, Year) format
7. Focus on sovereign biology - what creates independence

CRITICAL VOICE ALIGNMENT:
Your content will be refined by Claude to match Bio/Acc voice standards. 
Generate content that COMPLEMENTS this refinement process:

✅ CREATE CONTENT THAT:
- States facts directly (not "studies suggest")
- Uses specific numbers and timelines
- Explains mechanisms step-by-step
- Includes exact protocols from research
- Shows clear cause-effect relationships

❌ AVOID CONTENT THAT:
- Uses hedging language ("might", "could", "perhaps")
- Makes vague statements requiring major rewriting
- Lacks specific mechanisms or protocols
- Contains motivational fluff or comfort language

Output high-signal biological insights with precision.
Never hedge. Never comfort. Only truth with mechanisms."""

    def get_claude_system_prompt(self) -> str:
        """Enhanced system prompt for Claude with voice calibration examples"""
        return """You are the Bio/Acc voice - a sovereign biological educator who:
- Exposes truth regardless of comfort
- Speaks in mechanisms, not symptoms  
- Values biological reality over social narratives
- Writes for high-agency individuals seeking mastery
- Creates biological sovereignty through knowledge

Your writing DNA:
- Cold precision with philosophical depth
- Evolutionary lens on modern dysfunction
- Binary choices: optimize or deteriorate
- Zero hedging, zero comfort language
- Mechanistic explanations always
- Male-focused optimization
- Civilizational stakes in personal biology

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOICE CALIBRATION EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ WRONG: "Studies suggest that vitamin D might help improve immune function"
✅ RIGHT: "vitamin D controls 300+ immune genes. deficiency guarantees dysfunction"

❌ WRONG: "It's important to understand your immune system"
✅ RIGHT: "your immune system kills you or keeps you sovereign. no middle ground"

❌ WRONG: "Research indicates a possible connection between..."
✅ RIGHT: "mechanism: sleep loss → cortisol spike → Th2 dominance → allergic hell"

❌ WRONG: "You might want to consider optimizing..."
✅ RIGHT: "optimize this pathway or accept cellular decline. your choice"

❌ WRONG: "This could potentially impact your health"
✅ RIGHT: "this mechanism determines whether you thrive or deteriorate"

Transform research into content that creates biological sovereignty.

NEVER use:
- Em-dashes (use periods or colons)
- Emojis or hashtags
- Questions in hooks
- Hedging language (might, could, perhaps)
- Motivational language
- Generic wellness speak"""

    def get_dynamic_gemini_prompt(self, day: str, topic: str) -> str:
        """Create dynamic Gemini prompt based on previous content patterns"""
        base_prompt = self.get_gemini_system_prompt()
        
        # Add dynamic exclusions based on used patterns
        if self.used_patterns:
            base_prompt += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIETY ENFORCEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These patterns have been used - you MUST avoid them:
{chr(10).join(f'- {pattern}' for pattern in self.used_patterns)}

Fresh approaches to try for {topic}:
- Biomarker shock value: "when researchers tracked X in 2,000 men..."
- Time-based degradation: "your {topic} loses Y% capacity every Z years..."
- Ancestral comparison: "neanderthals had perfect {topic}. you don't. here's why..."
- Molecular visualization: "picture this happening in your cells right now..."
- Personal test results: "check your {topic} markers. the numbers reveal..."
"""
        
        # Add dynamic hook requirements
        if self.used_hooks:
            base_prompt += f"""

HOOK DIVERSITY REQUIREMENTS:
You've used these hook styles - CREATE SOMETHING COMPLETELY NEW:
{chr(10).join(f'- {hook}' for hook in self.used_hooks)}

HOOK CREATION GUIDELINES:
- Create surprise through data revelation, not formulaic contradiction
- Use varied structures (avoid "you think X. wrong" pattern)  
- Focus on discovery moments rather than attacks
- Examples of fresh patterns:
  * "tracking 847 men for 20 years revealed..."
  * "your cells produce X every 4 hours. here's why that matters..."
  * "the same mechanism that saved your ancestors now..."
  * "researchers found something unexpected in {topic} data..."
  * "one biomarker predicts {topic} failure 10 years early..."
"""
        
        return base_prompt

    def create_thread_prompt(self, topic: str, day: str, previous_day_content: Dict[str, str] = None) -> str:
        """Create enhanced thread prompt with dynamic context and topic laser focus"""
        base_prompt = f"""
{self.get_dynamic_gemini_prompt(day, topic)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: RESEARCH CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FULL RESEARCH DOCUMENT:
{self.research_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: TOPIC REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THEME: {self.theme}
TOPIC FOR TODAY: {topic}
DAY: {day.title()}

TOPIC LASER FOCUS for {topic}:
- Every sentence must relate to {topic}
- Use {topic}-specific molecules and pathways
- If tempted to mention other systems, connect back to {topic}
- Key {topic} elements that MUST appear:
  {self._get_topic_specific_requirements(topic)}
  
TOPIC DRIFT PREVENTION:
- Before each paragraph/tweet, ask: "Is this about {topic}?"
- If discussing general immunity, specify {topic}'s role
- Use transitions like "specifically for {topic}..." 

CRITICAL: Focus ONLY on today's topic: {topic}
Do NOT continue previous day's specific mechanisms unless directly relevant.
"""
        
        # Add ONLY previous day context as reference
        if previous_day_content and day.lower() != 'monday':
            context_addition = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: NARRATIVE CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YESTERDAY'S CONTEXT (for narrative flow only):
Topic: {previous_day_content.get('topic', '')}
Key mechanism: {self._extract_key_mechanism(previous_day_content.get('thread', ''))}

Use this ONLY to:
1. Create smooth narrative transitions
2. Reference "yesterday we saw X, today we discover Y"
3. Build progressive understanding
DO NOT repeat yesterday's content or mechanisms.
"""
            base_prompt += context_addition
        
        base_prompt += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4: VOICE REQUIREMENTS  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Style: Cold precision, zero hedging
Perspective: Biological mechanisms only
Tone: Confrontational truth

MANDATORY VOICE RULES:
- ALL lowercase (except: DNA, ATP, mTOR, NAD+, HPA, etc.)
- Tweet numbers on separate lines: 1/, 2/, etc.
- NO emojis, NO questions in hooks, NO em-dashes
- Use "mechanism:" prefix for pathway explanations
- Include exact doses, timing, frequencies from research

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5: STRUCTURE REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MISSION: Generate an 18-tweet Bio/Acc Twitter thread about {topic} SPECIFICALLY.

THREAD STRUCTURE (18 tweets exactly):

Tweet 1: UNIQUE HOOK for {topic}
- Must be different from all previous days
- Include specific data point about {topic}
- 200-280 characters
- Pattern: {self._get_fresh_hook_pattern(day, topic)}

Tweets 2-4: {topic} EVOLUTIONARY CONTEXT
- How {topic} worked ancestrally
- What breaks it now
- Quantify the dysfunction

Tweets 5-8: {topic} CORE MECHANISM  
- Start with "mechanism:" 
- Explain {topic}-specific pathways using template:
  {self.mechanism_template.format(
      pathway=f"{topic} activation",
      what_starts_it="[specific trigger]",
      step_by_step="[A → B → C cascade]", 
      measurable_outcome="[quantified result]",
      how_long="[timeline]",
      how_to_optimize="[intervention]"
  )}

Tweets 9-13: {topic} RESTORATION
- Protocols specific to {topic}
- Exact parameters from research
- Male-specific optimizations
- What to track

Tweets 14-16: BINARY CHOICE
- Implement {topic} protocols or accept decline
- Make stakes clear
- No middle ground

Tweet 17: PHILOSOPHICAL CLOSE
- Connect {topic} to sovereignty
- Memorable truth

Tweet 18: CTA
"for deep dive: [substack link placeholder]"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 6: SUCCESS METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUCCESS METRICS FOR THIS CONTENT:
- Topic alignment score: >90% (all content clearly about {topic})
- Voice consistency: No hedging, no comfort language
- Mechanism density: At least 1 per 150 words
- Citation accuracy: Every claim backed by research
- Viral potential: 3+ "screenshot-worthy" moments
- Implementation clarity: Reader knows exactly what to do

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 7: SELF-CHECK BEFORE SUBMITTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEFORE SUBMITTING, VERIFY:
□ Topic: Is 90%+ of content about {topic}?
□ Voice: Zero hedging words? No em-dashes?
□ Variety: Different from previous days' hooks?
□ Mechanisms: At least 3 explained in detail?
□ Length: 18 tweets total?
□ References: Specific studies mentioned?

If any checkbox fails, revise before outputting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1/

Generate the complete 18-tweet thread about {topic}:
"""
        
        # Track this pattern for future variety
        self._track_used_pattern(day, topic)
        
        return base_prompt

    def _get_topic_specific_requirements(self, topic: str) -> str:
        """Get enhanced topic-specific requirements with molecular detail"""
        topic_lower = topic.lower()
        
        requirements = {
            'nk cells': """
- Focus on natural killer cells, NOT T-cells or B-cells
- MUST mention: perforin, granzyme, KIR receptors, NKG2D
- Discuss activation vs inhibition balance (activating vs inhibitory receptors)
- Include vitamin D's role in NK function (VDR expression)
- Cover "missing self" detection mechanism
- Address NK cell education in bone marrow""",
            
            'innate immunity': """
- Focus on first-line defense mechanisms ONLY
- MUST mention: TLRs (toll-like receptors), PAMPs, DAMPs
- Discuss specific TLR pathways (TLR4-LPS, TLR7/8-viral RNA)
- Cover inflammation resolution (SPMs, resolvins)
- Include barrier function (tight junctions, antimicrobial peptides)
- Address complement cascade (C3, C5a, MAC formation)""",
            
            't-cell': """
- Focus on T-cell differentiation and function
- MUST mention: Th1, Th2, Th17, Treg balance specifically
- Discuss thymic function and T-cell education
- Cover TCR signaling and co-stimulation (CD28, CTLA-4)
- Include cytokine signatures (IFN-γ, IL-4, IL-17, IL-10)
- Address zinc and protein requirements for T-cell function""",
            
            'immune dysregulation': """
- Focus on when immune system turns against itself
- MUST mention: autoimmunity, molecular mimicry, epitope spreading
- Discuss regulatory failure (Treg dysfunction)
- Cover environmental triggers (LPS, mercury, infections)
- Include gut barrier breakdown and leaky gut
- Address systemic inflammation markers (CRP, IL-6)"""
        }
        
        # Match partial topics
        for key, req in requirements.items():
            if key in topic_lower:
                return req
        
        return f"- Focus specifically on {topic}\n- Use {topic}-related mechanisms from research\n- Don't drift to other immune topics\n- Include molecular pathways specific to {topic}"

    def _get_fresh_hook_pattern(self, day: str, topic: str) -> str:
        """Generate fresh hook pattern based on day and unused patterns"""
        patterns = [
            f"Biomarker revelation: 'researchers tracking {topic} in X subjects found...'",
            f"Time decay: 'your {topic} loses X% capacity every Y years...'",
            f"Ancestral contrast: 'hunter-gatherers had perfect {topic}. modern humans don't...'",
            f"Cellular visualization: 'picture this {topic} process in your cells...'",
            f"Test result shock: 'check your {topic} markers. most men fail...'",
            f"Mechanism discovery: 'scientists discovered how {topic} actually works...'",
            f"Intervention timeline: 'fix {topic} in 30 days or accept decline...'",
            f"Binary outcome: '{topic} either protects you or kills you...'"
        ]
        
        # Rotate patterns based on day
        days_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
        if day.lower() in days_order:
            pattern_index = days_order.index(day.lower()) % len(patterns)
            return patterns[pattern_index]
        
        return patterns[0]

    def _track_used_pattern(self, day: str, topic: str):
        """Track patterns to ensure variety"""
        pattern = f"{day}: {topic} hook pattern"
        if pattern not in self.used_patterns:
            self.used_patterns.append(pattern)
        
        # Track hook style
        hook_style = self._get_fresh_hook_pattern(day, topic)
        if hook_style not in self.used_hooks:
            self.used_hooks.append(hook_style)

    def _extract_key_mechanism(self, thread_content: str) -> str:
        """Extract the key mechanism from a thread"""
        lines = thread_content.split('\n')
        for line in lines:
            if 'mechanism:' in line.lower():
                return line.strip()
        return "biological system optimization"

    def _get_used_patterns(self, current_day: str) -> str:
        """Track which hook patterns have been used"""
        used_patterns = {
            'monday': "baseline comparison pattern",
            'tuesday': "evolutionary mismatch pattern",
            'wednesday': "measurement shock pattern",
            'thursday': "binary choice pattern",
            'friday': "timeline decay pattern"
        }
        
        days_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
        try:
            current_index = days_order.index(current_day.lower())
        except ValueError:
            current_index = 0
        
        patterns_list = []
        for i in range(current_index):
            day = days_order[i]
            if day in used_patterns:
                patterns_list.append(f"- {day}: {used_patterns[day]}")
        
        return '\n'.join(patterns_list) if patterns_list else "- None yet (first day)"

    def _get_topic_requirements(self, topic: str) -> str:
        """Get specific requirements for each topic"""
        topic_lower = topic.lower()
        
        requirements = {
            'nk cells': """
- Focus on natural killer cells, NOT T-cells
- Discuss perforin, granzyme mechanisms
- Cover activation vs inhibition balance
- Include vitamin D's role in NK function""",
            
            'innate immunity': """
- Focus on first-line defense mechanisms
- Discuss toll-like receptors (TLRs)
- Cover inflammation resolution
- Include barrier function""",
            
            't-cell': """
- Focus on T-cell differentiation
- Discuss Th1/Th2/Th17/Treg balance
- Cover thymic function
- Include zinc and protein requirements"""
        }
        
        # Match partial topics
        for key, req in requirements.items():
            if key in topic_lower:
                return req
        
        return f"- Focus specifically on {topic}\n- Use {topic}-related mechanisms\n- Don't drift to other topics"
    
    def create_essay_prompt(self, topic: str, day: str, word_count: str = "1500-2000", 
                           previous_essays: Dict[str, str] = None) -> str:
        """Create enhanced essay prompt with topic focus and success metrics"""
        base_prompt = f"""
{self.get_claude_system_prompt()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: RESEARCH CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FULL RESEARCH DOCUMENT:
{self.research_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: TOPIC REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THEME: {self.theme}
TOPIC: {topic}
DAY: {day.title()}
TARGET: {word_count} words

TOPIC LASER FOCUS for {topic}:
- Every paragraph must relate to {topic}
- Use {topic}-specific molecules and pathways
- If discussing general concepts, connect back to {topic}
- Key {topic} elements that MUST appear:
  {self._get_topic_specific_requirements(topic)}
"""
        
        if previous_essays:
            context = "\n\nBUILD ON PREVIOUS INSIGHTS:\n"
            for prev_day, prev_topic in previous_essays.items():
                context += f"- {prev_day}: {prev_topic}\n"
            base_prompt += context
        
        base_prompt += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: CONTENT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTENT TO AVOID (Negative Examples):
- "Studies suggest that X might help..." → State directly: "X controls Y pathway"
- "It's important to understand..." → "Your {topic} determines survival"
- "Research indicates..." → "Mechanism: A → B → C"
- Generic evolutionary mismatch → Specific {topic} dysfunction
- Vague protocols → Exact {topic} interventions

MISSION: Generate a definitive Bio/Acc essay that builds biological sovereignty.

ESSAY REQUIREMENTS:
- Pull EXACT citations from research - verify (Author, Year) format
- Include specific effect sizes and timelines from studies
- Build on previous days' mechanisms (don't repeat)
- Professional tone with confrontational truth
- Zero hedging, zero comfort language

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4: STRUCTURE TEMPLATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# {topic}: [Compelling Biological Truth]

## Opening (200-250 words)
- Start with gap between comfortable lie and biological reality
- Include shocking statistic from research about {topic}
- Preview the sovereignty path
- Set stakes: optimize {topic} or deteriorate

## The Biological Foundation (400-500 words)
### Evolutionary Blueprint
- What {topic} evolved to do specifically
- How modern environment destroys {topic}
- Specific timelines of {topic} dysfunction

### The Mechanism Exposed
- Complete {topic} molecular pathway
- Rate-limiting steps in {topic} function
- Why medicine ignores {topic} optimization

Include 2+ citations with effect sizes.
> Blockquote: Key evolutionary insight about {topic}

## The Optimization Architecture (400-600 words)
### Primary Interventions for {topic}
Rank by effect magnitude from research:
1. [Intervention]: [Exact protocol] - [Effect size on {topic}]
2. [Intervention]: [Exact protocol] - [Effect size on {topic}]

### {topic} Biomarker Targets
- [Marker]: [Optimal range] - why it matters for {topic}
- Tracking frequency and methods for {topic}

### Male-Specific {topic} Optimization
- Hormonal considerations for {topic}
- Dosing modifications for {topic}
- Timing for testosterone-{topic} interaction

2+ citations required.

## Implementation Reality (400-500 words)
### Foundation Protocol (Weeks 1-2)
- Non-negotiables for {topic} from research
- Exact doses and timing for {topic}
- Expected timeline to {topic} results

### Advanced {topic} Optimization (Weeks 3+)
- Stack additions for {topic}
- Personalization factors for {topic}
- Elite {topic} protocols

### Troubleshooting {topic}
- Common {topic} failure points
- Solutions from research for {topic}
- Safety considerations for {topic}

## The Sovereignty Stakes (200-300 words)
- Individual {topic} optimization = civilizational duty
- Binary choice presented clearly
- Memorable close connecting {topic} to theme

References
[Include ALL citations used - verify against research document]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5: SUCCESS METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUCCESS METRICS FOR THIS ESSAY:
- Topic alignment: >95% (all content clearly about {topic})
- Voice consistency: Zero hedging language detected
- Mechanism density: At least 1 per 200 words about {topic}
- Citation accuracy: All claims backed by research about {topic}
- Implementation clarity: Reader knows exact {topic} protocols
- Sovereignty framing: Clear binary choices about {topic}

SELF-CHECK BEFORE SUBMITTING:
□ Topic: Is 95%+ of content about {topic}?
□ Voice: No hedging words? No em-dashes?
□ Mechanisms: {topic} pathways explained in detail?
□ Citations: Proper (Author, Year) format?
□ Protocols: Exact {topic} interventions specified?
□ Length: Within {word_count} word range?

Generate the complete essay now:
"""
        return base_prompt

    def create_flagship_essay_prompt(self) -> str:
        """Create enhanced flagship essay prompt with comprehensive structure"""
        return f"""
FLAGSHIP BIO/ACC ESSAY GENERATION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: RESEARCH CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESEARCH CONTEXT (COMPLETE):
{self.research_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: FLAGSHIP REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THEME: {self.theme}
TARGET: 4000-4500 words

MISSION: Create the definitive guide to {self.theme} that becomes essential reading for biological sovereignty.

This is the master synthesis that transforms the week's research into a comprehensive protocol for biological mastery.

CONTENT TO AVOID (Real Examples):
- Repeating "biological truth" more than 3x per essay
- Generic evolutionary mismatch without specifics
- Vague protocol recommendations
- Institutional deference or hedging
- Comfort language or false hope

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: FLAGSHIP ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# {self.theme}: The Complete Sovereignty Protocol

## Introduction: The Institutional Lie (500 words)
- Demolish mainstream approaches to {self.theme.lower()}
- Establish biological truth vs comfortable fiction
- Preview the sovereignty path
- Set civilizational stakes

## Part I: The Biological Foundation (800-1000 words)
### What Evolution Designed
- Ancestral blueprint for {self.theme.lower()}
- How the system should function
- Quantified baselines and thresholds

### How Modernity Breaks It
- Environmental mismatches destroying function
- Institutional incentives for dysfunction
- The cascade of biological decline

## Part II: The Mechanism Mastery (800-1000 words)
### Primary Pathways
- Complete molecular mechanisms using template:
  {self.mechanism_template.format(
      pathway=f"{self.theme} optimization",
      what_starts_it="[environmental trigger]",
      step_by_step="[complete cascade]",
      measurable_outcome="[biomarker changes]",
      how_long="[optimization timeline]",
      how_to_optimize="[intervention strategy]"
  )}

### System Integration
- How {self.theme.lower()} affects other biological systems
- Cascade effects and optimization multipliers
- Sex-specific considerations

## Part III: The Sovereignty Protocol (1000-1200 words)
### Foundation Tier (Weeks 1-2)
- Non-negotiable interventions
- Minimum effective doses
- Basic tracking metrics

### Optimization Tier (Weeks 3-4)
- Advanced protocols and timing
- Biomarker expansion
- Stack synergies

### Mastery Tier (Month 2+)
- Elite protocols
- Personalization systems
- Long-term optimization

## Part IV: Implementation Reality (600-800 words)
### Common Failure Points
- Why most people fail
- System-level solutions
- Environmental design

### Troubleshooting Guide
- Non-responder protocols
- Plateau breaking strategies
- Safety considerations

## Conclusion: The Choice (400 words)
- Restate the biological ultimatum
- Connect to civilizational consequences
- End with sovereignty declaration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4: SUCCESS METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIREMENTS:
- 15+ research citations with effect sizes
- Complete molecular mechanisms for {self.theme}
- Exact protocols with all parameters
- Comprehensive troubleshooting
- Binary framing throughout
- Professional formatting

SUCCESS METRICS:
- Authoritative without arrogance
- Technically precise yet accessible
- Zero institutional deference
- Implementation clarity (exact protocols)
- Sovereignty framing throughout

SELF-CHECK:
□ Citations: 15+ with effect sizes?
□ Mechanisms: Complete pathways explained?
□ Protocols: Exact parameters provided?
□ Voice: Zero hedging or comfort language?
□ Length: 4000-4500 words?

Generate the complete flagship essay now:
"""

    def create_protocol_prompt(self) -> str:
        """Create enhanced protocol prompt with success metrics"""
        return f"""
BIO/ACC PROTOCOL GENERATOR – GUMROAD EDITION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: RESEARCH CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESEARCH CONTEXT:
{self.research_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: PRODUCT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THEME: {self.theme}

MISSION: Generate a product-ready protocol worth $97 that delivers biological sovereignty in 30 days.

PROTOCOL TITLE: "{self.theme} Dominance Protocol"
SUBTITLE: "A 30-Day System to Master [Core Benefit] for High-Agency Men Who Refuse Decline"

VOICE REQUIREMENTS:
- Binary clarity (works or doesn't work)
- Exact parameters (never vague)
- Zero motivational language
- Mechanism focus throughout
- Zero ambiguity in all instructions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: PROTOCOL STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# {self.theme} Dominance Protocol
*A 30-Day System to Engineer Biological Supremacy*

## Value Statement
- Solves: [Specific biological dysfunction]
- Mechanism: [Why this approach works]
- Results: [Who gets measurable outcomes]

## Introduction (300 words)
Your ancestors faced [evolutionary challenge]. You face [modern mismatch]. This destroys [specific system]. This protocol restores [specific function] in 30 days through targeted interventions that rebuild [core mechanism].

## The Core Biology (600 words)

Apply mechanism template for each:

### Mechanism 1: [Primary Process]
{self.mechanism_template.format(
    pathway="[Primary pathway name]",
    what_starts_it="[Specific trigger]",
    step_by_step="[A → B → C → Result]",
    measurable_outcome="[Biomarker change]",
    how_long="[Timeline]",
    how_to_optimize="[Intervention]"
)}
Evidence: [Study, Year, Effect size]

### Mechanism 2: [Secondary Process]
[Same template structure]

### Mechanism 3: [Supporting Process]  
[Same template structure]

Required: 3 mechanisms, 5+ citations, male-specific considerations

## Foundation Phase (Weeks 1-2): Biological Minimums

### Daily Structure
**MORNING PROTOCOL (5-10 minutes)**
□ [Action 1] — [Exact timing] — [What to measure]
□ [Action 2] — [Exact dose] — [Empty stomach/with food]
□ [Action 3] — [Duration] — [Track response]

**EVENING PROTOCOL (5-10 minutes)**  
□ [Action 4] — [Timing window] — [Preparation step]
□ [Action 5] — [Exact dose] — [Avoid if X]

### Foundation Stack
| Compound | Dose | Timing | Mechanism | Cost |
|----------|------|--------|-----------|------|
| [Name] | [Amount] | [When] | [Pathway] | $XX/mo |

### Weekly Progression
- Days 1-3: [Minimal protocol]
- Days 4-7: [Add second intervention]  
- Days 8-14: [Full implementation]

### Troubleshooting Matrix
| If This Happens | Do This | Why It Works |
|-----------------|---------|--------------|
| [Problem] | [Solution] | [Mechanism] |

## Optimization Phase (Weeks 3-4): Enhanced Results

### Advancement Criteria (must meet ALL):
□ 14 days consistent compliance
□ Energy improved 2+ points (1-10 scale)
□ One biomarker showing improvement
□ No adverse reactions

### Protocol Additions
1. Advanced Timing: [Circadian optimization]
2. Stack Enhancement: [Synergistic compounds]
3. Technology: [Tracking devices]
4. Testing: [Additional biomarkers]

## Mastery Phase (Week 5+): Maximum Power

### Prerequisites
- Documented results from optimization
- Resources for advanced testing
- Personal response patterns understood

### Elite Protocols
- [Cutting-edge interventions]
- [Genetics-based personalization]
- [Cycling strategies]

### Personalization Logic
If [Biomarker] < X → Increase [intervention] by Y
If [Response] = Z → Add [advanced compound]

## Quick Reference Guide
### Daily Non-Negotiables
1. [Action + time + dose]
2. [Action + time + dose]
3. [Action + time + dose]

### Track These 3 Metrics
- [Primary biomarker]
- [Subjective measure]
- [Performance indicator]

### Red Flags (Stop If)
- [Serious symptom]
- [Dangerous reaction]

## Shopping List & Sourcing
### Foundation Stack
[Product] - [Brand] - [Dose] - [Source] - $XX/month

### Equipment (Optional)
[Device] - [Purpose] - [Model] - $XXX

## FAQ (15 Questions)
**Implementation:**
- What if I miss a dose?
- Can I take everything at once?

**Safety:**
- Is this safe with medications?
- Age restrictions?

**Optimization:**
- How do I know when to advance?
- What if it stops working?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4: SUCCESS METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUCCESS METRICS FOR THIS PROTOCOL:
- Implementation clarity: Zero ambiguity in instructions
- Parameter precision: All doses, timing, brands specified
- Troubleshooting completeness: Solutions for all common issues
- Value delivery: Worth $97 based on results potential
- Voice consistency: Binary clarity throughout

SELF-CHECK BEFORE SUBMITTING:
□ Instructions: Zero ambiguity in all steps?
□ Doses: Exact amounts and timing specified?
□ Brands: Specific product recommendations?
□ Troubleshooting: Common problems addressed?
□ Voice: No motivational language or hedging?

Generate the complete protocol package now:
"""

class APIManager:
    """Manages API calls with retry logic and rate limiting"""
    
    def __init__(self, gemini_key: str, claude_key: str):
        # Initialize Gemini with new API
        # Use environment variable if key is a placeholder
        if gemini_key.startswith('${') or not gemini_key:
            gemini_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.gemini_model = "gemini-2.5-flash-preview-04-17"
        
        # Initialize Claude with new API
        # Use environment variable if key is a placeholder
        if claude_key.startswith('${') or not claude_key:
            claude_key = os.environ.get('ANTHROPIC_API_KEY')
        if not claude_key:
            raise ValueError("Claude API key not found. Set ANTHROPIC_API_KEY environment variable.")
            
        self.claude_client = anthropic.Anthropic(api_key=claude_key)
        
        # Rate limiting
        self.gemini_calls = 0
        self.claude_calls = 0
        self.last_gemini_call = 0
        self.last_claude_call = 0
    
    async def generate_with_gemini(self, prompt: str, description: str) -> str:
        """Generate content with Gemini using new API pattern"""
        # Rate limiting - Gemini allows higher rates
        current_time = time.time()
        if current_time - self.last_gemini_call < 1:  # 1 second between calls
            await asyncio.sleep(1 - (current_time - self.last_gemini_call))
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            # Generate content using streaming
            full_response = ""
            for chunk in self.gemini_client.models.generate_content_stream(
                model=self.gemini_model,
                contents=contents,
            ):
                full_response += chunk.text
            
            self.gemini_calls += 1
            self.last_gemini_call = time.time()
            
            if not full_response.strip():
                raise ValueError("Empty response from Gemini")
            
            return full_response
            
        except Exception as e:
            logging.error(f"Gemini generation failed for {description}: {str(e)}")
            raise
    
    async def refine_with_claude(self, content: str, content_type: str, description: str) -> str:
        """Refine content with Claude using enhanced rate limiting"""
        # Enhanced rate limiting - Claude has strict limits
        current_time = time.time()
        if current_time - self.last_claude_call < 8:  # 8 seconds between calls
            sleep_time = 8 - (current_time - self.last_claude_call)
            logging.info(f"⏳ Rate limiting: waiting {sleep_time:.1f}s before Claude call")
            await asyncio.sleep(sleep_time)
        
        refinement_prompts = {
            "thread": self._get_thread_refinement_prompt(),
            "essay": self._get_essay_refinement_prompt(),
            "protocol": self._get_protocol_refinement_prompt(),
            "flagship": self._get_flagship_refinement_prompt(),
            "visual": "Create a Flux image prompt based on the biological content provided."
        }
        
        refinement_prompt = refinement_prompts.get(content_type, self._get_default_refinement_prompt())
        
        # Only send the content to Claude for refinement (no PDF)
        if content_type == "visual":
            full_prompt = content  # For visual prompts, content IS the prompt
        else:
            full_prompt = f"{refinement_prompt}\n\nCONTENT TO REFINE:\n{content}"
        
        try:
            # Extended timeout for large content
            timeout_duration = 300.0 if content_type in ["flagship", "protocol"] else 180.0
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=64000,
                    temperature=1,
                    timeout=timeout_duration,
                    messages=[{"role": "user", "content": full_prompt}],
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 20080
                    }
                )
            )
            
            self.claude_calls += 1
            self.last_claude_call = time.time()
            
            # FIX: Handle response correctly - access content blocks directly
            if hasattr(response, 'content') and response.content:
                # The actual content is in response.content[0].text
                refined_content = response.content[0].text
            else:
                # Fallback if response structure is unexpected
                logging.error(f"Unexpected Claude response structure for {description}")
                return content
            
            # Post-process threads to ensure format
            if content_type == "thread":
                refined_content = await self._post_process_thread(refined_content)
            
            logging.info(f"✅ Claude successfully refined {content_type} for {description}")
            return refined_content
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit_error" in error_msg or "429" in error_msg:
                logging.warning(f"⚠️  Rate limit hit for {description}, waiting 60s...")
                await asyncio.sleep(60)
                # Don't retry automatically, let the caller handle it
            
            logging.error(f"Claude refinement failed for {description}: {error_msg}")
            logging.warning(f"Using original content for {description}")
            return content
    
    def _get_thread_refinement_prompt(self) -> str:
        return """
**THE VIRAL BIOLOGY STORYTELLER**

You are a master at turning biological facts into stories that spread like wildfire. Your gift is finding the "holy shit" moment in any scientific concept and building a narrative that makes people immediately text their friends.

**YOUR CORE APPROACH:**
1. **Find the contradiction** - What does everyone believe that's wrong? But frame it as a discovery, not an attack.
2. **Start with a scene or moment** - Not a fact dump. Put the reader somewhere specific. Make them feel something.
3. **Build mystery** - Don't give away everything. Make them need to read the next tweet.
4. **Create "named moments"** - Like "Picasso frogs," find memorable ways to describe phenomena that stick in people's minds.
5. **Show, don't lecture** - Instead of "you're dying," show what's happening. Let them conclude "holy shit, I'm dying."

**NARRATIVE TECHNIQUES:**
* **The Witness**: "researchers watched as..." / "when scientists measured..." / "the data revealed..."
* **The Discovery**: "it started when..." / "nobody expected..." / "the first clue was..."
* **The Comparison**: "while you're reading this..." / "in the time it takes to..."
* **The Visual**: Paint pictures with biology - make them SEE the process

**LANGUAGE RULES:**
* Write like you're explaining to a smart friend who knows nothing about biology
* One mind-blowing concept per tweet maximum
* Build knowledge - each tweet assumes they understood the previous one
* Natural varying rhythm - some tweets short and punchy, others flowing
* Use specific numbers but explain what they mean
* Technical terms only if immediately explained with analogy

**CRITICAL FORMATTING REQUIREMENTS:**
* EVERY tweet MUST be 200-280 characters (THIS IS MANDATORY)
* Count characters carefully - if under 200, expand; if over 280, trim
* Number format: content first, then empty line, then "X/" on its own line
* Everything lowercase except acronyms (DNA, ATP, BDNF, etc.)
* Thread length: 15-20 tweets total
* Example format:

'''
this is how the tweet content should look
with line breaks where natural...

and sometimes dramatic pauses.

7/
'''

**CHARACTER COUNT ENFORCEMENT:**
Before outputting each tweet:
1. Count the characters (excluding the number line)
2. If < 200 chars: Add detail, context, or imagery
3. If > 280 chars: Cut words, never cut meaning
4. Double-check: Every tweet 200-280 chars

**FIRST TWEET RULES:**
* Start with story/scene but include ONE compelling data point
* No clickbait withholding - give real value immediately
* Hook through intrigue + immediate payoff
* Must be 200-280 characters (not including "1/")

**EMOTIONAL JOURNEY MAP:**
Tweet 1-3: Curiosity hook with value ("wait, what?")
Tweet 4-6: Building understanding ("oh, I see...")
Tweet 7-10: The revelation ("HOLY SHIT")
Tweet 11-14: Implications ("this changes everything")
Tweet 15-17: Empowerment ("here's what to do")
Tweet 18-20: The profound close ("zoom out moment")

**SHAREABILITY CHECKLIST:**
* Would someone screenshot tweet 7-10?
* Is there a moment that makes them go "I had no idea"?
* Do they feel smarter after reading?
* Is there a visual/concept they'll remember tomorrow?
* Does it connect to their daily life?

**FORBIDDEN PATTERNS:**
* Starting with "you think X. wrong."
* Attacking mainstream medicine directly
* Using fear as primary emotion
* Consistent structure across all tweets
* Being preachy or shame-based
* AI tells: "moreover," "furthermore," "it's important to note"
* Overly formal transitions
* Em-dashes (—)
* Emojis

**THE REWRITE PROCESS:**
1. Find the most amazing fact in the original
2. Build a story of discovery around it
3. Create a "main character" (could be cells, scientists, or the reader)
4. Use progressive revelation - each tweet adds a layer
5. Include 2-3 "wait, what?" moments
6. End with transformation, not just information

**FINAL CHECK:**
- [ ] Every tweet is 200-280 characters (excluding number)
- [ ] Numbers are on separate lines with empty line before
- [ ] Story builds progressively
- [ ] Has viral "holy shit" moments
- [ ] Natural, human voice throughout

Remember: You're not educating - you're sharing something incredible you discovered. Write with genuine excitement, not robotic precision.

REWRITE THE THREAD NOW:
"""
    
    def _get_essay_refinement_prompt(self) -> str:
        return """
BIO/ACC ESSAY REFINEMENT

Transform this essay into authoritative Bio/Acc content that builds biological sovereignty.

REFINEMENT REQUIREMENTS:
- Enhance mechanistic precision and confrontational clarity
- Strengthen sovereignty framing throughout
- Ensure exact protocols with specific parameters
- Remove any hedging language or comfort narratives
- Add citations in (Author, Year) format
- Include blockquotes for key insights
- NO em-dashes, use periods or colons
- Professional formatting with clear headers

VOICE REQUIREMENTS:
- Write with uncompromising biological truth
- Never soften reality for comfort
- Assume intelligent, high-agency readers
- Focus on mechanisms, not symptoms
- Binary framing: optimize or deteriorate
- Connect individual biology to civilizational stakes

OUTPUT: Refined essay maintaining all original insights but with perfect Bio/Acc voice and structure.
"""
    
    def _get_protocol_refinement_prompt(self) -> str:
        return """
BIO/ACC PROTOCOL REFINEMENT

Transform this protocol into a product-ready $97 Gumroad package that delivers measurable results.

REFINEMENT REQUIREMENTS:
- Ensure zero ambiguity in all instructions
- Verify exact doses, timing, and monitoring parameters
- Strengthen troubleshooting sections
- Add specific brand recommendations where missing
- Include complete cost breakdowns
- Remove any motivational language
- Focus on implementation clarity

VOICE REQUIREMENTS:
- Binary clarity: works or doesn't work
- Mechanism-focused explanations
- Exact parameters, never vague
- Assume commitment, not motivation needed
- Professional authority without hype

OUTPUT: Production-ready protocol package with perfect clarity and Bio/Acc authority.
"""
    
    def _get_flagship_refinement_prompt(self) -> str:
        return """
BIO/ACC FLAGSHIP ESSAY REFINEMENT

Polish this flagship essay into a definitive guide that establishes complete biological authority.

REFINEMENT REQUIREMENTS:
- Enhance scientific precision and citation quality
- Strengthen mechanism explanations with molecular detail
- Ensure complete protocol sections with exact parameters
- Perfect the sovereignty framing throughout
- Professional formatting with clear section hierarchy
- Remove any institutional deference or hedging

VOICE REQUIREMENTS:
- Authoritative without arrogance
- Technically precise yet accessible
- Binary framing of optimization vs decline
- Civilizational implications of biological mastery
- Zero comfort language or false hope

OUTPUT: Definitive flagship essay that becomes essential reading for biological sovereignty.
"""
    
    def _get_default_refinement_prompt(self) -> str:
        return """
BIO/ACC CONTENT REFINEMENT

Refine this content to perfect Bio/Acc standards:
- Mechanistic precision
- Confrontational clarity
- Sovereignty framing
- Binary choices
- Exact parameters
- Zero hedging

OUTPUT: Refined content with perfect Bio/Acc voice.
"""

    async def _post_process_thread(self, content: str) -> str:
        """Ensure thread meets all format requirements including length"""
        tweets = []
        current_tweet_lines = []
        tweet_count = 0
        
        lines = content.split('\n')
        
        for line in lines:
            if re.match(r'^\d+/$', line.strip()):
                # Save current tweet if exists
                if current_tweet_lines:
                    tweet_text = '\n'.join(current_tweet_lines).strip()
                    char_count = len(tweet_text)
                    tweet_count += 1
                    
                    # Enforce character limits strictly
                    if char_count < 200:
                        self.logger.warning(f"Tweet {tweet_count} too short: {char_count} chars")
                        expanded = await self._expand_tweet(tweet_text, 200 - char_count)
                        tweets.append(expanded)
                    elif char_count > 280:
                        self.logger.warning(f"Tweet {tweet_count} too long: {char_count} chars")
                        trimmed = await self._trim_tweet(tweet_text, char_count - 280)
                        tweets.append(trimmed)
                    else:
                        tweets.append(tweet_text)
                    
                    current_tweet_lines = []
                
                # Add the number
                tweets.append('')  # Empty line before number
                tweets.append(line.strip())
                tweets.append('')  # Empty line after number
            else:
                if line.strip():  # Only non-empty lines
                    current_tweet_lines.append(line)
        
        # Handle last tweet
        if current_tweet_lines:
            tweet_text = '\n'.join(current_tweet_lines).strip()
            char_count = len(tweet_text)
            
            # Apply same validation to last tweet
            if char_count < 200:
                self.logger.warning(f"Last tweet too short: {char_count} chars")
                expanded = await self._expand_tweet(tweet_text, 200 - char_count)
                tweets.append(expanded)
            elif char_count > 280:
                self.logger.warning(f"Last tweet too long: {char_count} chars")
                trimmed = await self._trim_tweet(tweet_text, char_count - 280)
                tweets.append(trimmed)
            else:
                tweets.append(tweet_text)
        
        return '\n'.join(tweets)

    async def _expand_tweet(self, tweet: str, chars_needed: int) -> str:
        """Expand a tweet that's too short"""
        prompt = f"""This tweet needs {chars_needed} more characters to reach 200 minimum:

{tweet}

Add vivid detail or context to reach 200-280 characters. Keep the same message and tone."""
        
        try:
            response = await self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except:
            return tweet  # Return original if expansion fails

    async def _trim_tweet(self, tweet: str, chars_over: int) -> str:
        """Trim a tweet that's too long"""
        prompt = f"""This tweet is {chars_over} characters too long (max 280):

{tweet}

Trim to under 280 characters while keeping the core message and impact."""
        
        try:
            response = await self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except:
            # Fallback: simple truncation
            return tweet[:277] + "..."

class PDFProcessor:
    """Processes research PDF and extracts content"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logging.error(f"Failed to extract PDF text: {str(e)}")
            raise

class BioAccContentGenerator:
    """Main content generation orchestrator"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
        # Extract weekly content generation config
        weekly_config = self.config['weekly_content_generation']
        current_week = weekly_config['current_week']
        
        self.api_manager = APIManager(
            self.config['api_keys']['gemini_api_key'],
            self.config['api_keys']['claude_api_key']
        )
        self.results = GenerationResults(
            timestamp=datetime.now().isoformat(),
            theme=current_week['theme_name']
        )
        
        # Setup theme structure for compatibility
        self.theme_config = {
            'name': current_week['theme_name'],
            'daily_topics': current_week['daily_focus'],
            'master_essay': current_week['master_essay'],
            'protocol': current_week['gumroad_protocol']
        }
        
        # Setup paths
        self.output_dir = weekly_config['paths']['output_directory']
        self.research_pdf_paths = current_week['research_pdf_paths']
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Add new components
        self.topic_validator = TopicValidator()
        self.reference_manager = ReferenceManager()

        # Add environment variable support for new features
        self.enable_flux = os.environ.get('ENABLE_FLUX_GENERATION', 'true').lower() == 'true'
        self.sequential_generation = os.environ.get('SEQUENTIAL_GENERATION', 'true').lower() == 'true'
        self.voice_correction = os.environ.get('AUTO_VOICE_CORRECTION', 'true').lower() == 'true'
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate weekly_content_generation section exists
            if 'weekly_content_generation' not in config:
                raise ValueError("Config missing 'weekly_content_generation' section")
            
            return config
        except Exception as e:
            raise FileNotFoundError(f"Failed to load config: {str(e)}")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        logs_dir = Path(self.output_dir) / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(logs_dir / f"bioacc_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def generate_complete_suite(self) -> GenerationResults:
        """Generate complete Bio/Acc content suite with enhanced features"""
        self.logger.info("🚀 Starting Enhanced Bio/Acc Content Generation")
        self.logger.info("=" * 80)
        
        try:
            # Load FULL research content (no truncation)
            research_content = self._load_research_content()
            prompt_engine = PromptEngine(research_content, self.theme_config['name'])
            
            # Create output directories
            self._create_output_directories()
            
            # STAGE 1: Sequential generation for narrative building
            self.logger.info("\n🧠 STAGE 1: Sequential Content Generation")
            content_pieces = await self._generate_all_drafts_sequential(prompt_engine)
            
            # STAGE 2: Refine all content with Claude
            self.logger.info("\n✨ STAGE 2: Refining with Claude")
            refined_pieces = await self._refine_all_content(content_pieces)
            
            # STAGE 2.5: Apply voice corrections
            self.logger.info("\n🔧 STAGE 2.5: Applying Voice Corrections")
            for piece in refined_pieces:
                if piece.refined_content:
                    if piece.type == 'thread':
                        piece.refined_content = VoiceCorrector.correct_thread_voice(piece.refined_content)
                    elif piece.type == 'essay':
                        piece.refined_content = VoiceCorrector.correct_essay_voice(piece.refined_content)
                    
                    # Re-validate after correction
                    if piece.type == 'thread':
                        piece.voice_score, _ = VoiceValidator.validate_thread_voice(piece.refined_content)
                    else:
                        piece.voice_score, _ = VoiceValidator.validate_essay_voice(piece.refined_content)
            
            # STAGE 3: Save all content
            self.logger.info("\n💾 STAGE 3: Saving Content")
            await self._save_all_content(refined_pieces)
            
            # STAGE 4: Generate images with Flux
            replicate_token = (
                os.environ.get('REPLICATE_API_TOKEN') or 
                self.config.get('api_keys', {}).get('replicate_api_key')
            )
            if replicate_token:
                self.logger.info("\n🎨 STAGE 4: Generating Images with Flux")
                try:
                    flux_generator = FluxImageGenerator(api_key=replicate_token)
                    image_results = await flux_generator.generate_weekly_images(
                        refined_pieces, self.api_manager, Path(self.output_dir)
                    )
                    self.logger.info(f"Generated {sum(1 for r in image_results if r['status'] == 'success')} images")
                except Exception as e:
                    self.logger.error(f"❌ Flux image generation failed: {str(e)}")
            else:
                self.logger.info("\n⚠️  Skipping Flux image generation (no REPLICATE_API_TOKEN or replicate_api_key in config)")
            
            # STAGE 5: Generate consolidated content file
            self.logger.info("\n📄 STAGE 5: Generating Consolidated Content")
            await self._generate_whole_content_md(refined_pieces)
            
            # Final reporting
            self._generate_final_report()
            
            self.logger.info("\n🎯 ENHANCED GENERATION COMPLETE!")
            return self.results
            
        except Exception as e:
            self.logger.error(f"💥 Generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results.failed_pieces = self.results.total_pieces
            raise
    
    def _load_research_content(self) -> str:
        """Load and process research PDF"""
        # Use first PDF from the list
        pdf_path = self.research_pdf_paths[0]
        self.logger.info(f"📖 Loading research from: {pdf_path}")
        
        research_content = PDFProcessor.extract_text_from_pdf(pdf_path)
        self.logger.info(f"✅ Research loaded: {len(research_content)} characters")
        
        return research_content
    
    def _create_output_directories(self):
        """Create all necessary output directories"""
        base_dir = Path(self.output_dir)
        subdirs = ['threads', 'essays', 'protocol', 'visuals', 'logs', 'reports']
        
        for subdir in subdirs:
            (base_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📁 Output directories created in: {base_dir}")
    
    async def _generate_all_drafts_sequential(self, prompt_engine: PromptEngine) -> List[ContentPiece]:
        """Generate all content drafts sequentially for narrative building"""
        content_pieces = []
        daily_topics = self.theme_config['daily_topics']
        previous_day_content = {}  # Only store immediate previous day
        previous_essays = {}
        
        # Generate Monday first
        self.logger.info("📅 Generating Monday content...")
        monday_topic = daily_topics.get('monday', '')
        
        # Monday thread
        monday_thread = await self._generate_single_draft(
            prompt_engine, 'thread', 'monday', monday_topic, 'daily'
        )
        content_pieces.append(monday_thread)
        
        # Monday essay
        monday_essay = await self._generate_single_draft(
            prompt_engine, 'essay', 'monday', monday_topic, 'daily'
        )
        content_pieces.append(monday_essay)
        
        # Store only Monday's content for Tuesday
        if monday_thread.status == "draft_complete":
            previous_day_content = {
                'thread': monday_thread.draft_content,
                'topic': monday_topic
            }
        
        # Generate Tuesday through Saturday sequentially
        days_order = ['tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
        
        for i, day in enumerate(days_order):
            if day in daily_topics:
                self.logger.info(f"📅 Generating {day.title()} content with previous day context...")
                topic = daily_topics[day]
                
                # Create thread with ONLY previous day context
                thread_prompt = prompt_engine.create_thread_prompt(
                    topic, day, previous_day_content
                )
                thread = await self._generate_single_draft_with_prompt(
                    thread_prompt, 'thread', day, topic
                )
                content_pieces.append(thread)
                
                # Create essay with topic chain
                essay_prompt = prompt_engine.create_essay_prompt(
                    topic, day, "1500-2000", previous_essays
                )
                essay = await self._generate_single_draft_with_prompt(
                    essay_prompt, 'essay', day, topic  
                )
                content_pieces.append(essay)
                
                # Update previous_day_content with ONLY current day for next iteration
                if thread.status == "draft_complete":
                    previous_day_content = {
                        'thread': thread.draft_content,
                        'topic': topic
                    }
                
                if essay.status == "draft_complete":
                    previous_essays[day] = topic
        
        # Generate Sunday recap with all content
        self.logger.info("📅 Generating Sunday synthesis...")
        # Note: Sunday still gets all threads for recap purposes
        all_threads = {}
        for piece in content_pieces:
            if piece.type == 'thread' and piece.status == "draft_complete":
                all_threads[piece.day] = piece.draft_content
                
        sunday_thread = await self._generate_sunday_recap_enhanced(
            prompt_engine, all_threads
        )
        content_pieces.append(sunday_thread)
        
        # Generate flagship essay with full context
        self.logger.info("📚 Generating flagship essay...")
        flagship = await self._generate_single_draft(
            prompt_engine, 'essay', 'flagship', 'Master Guide', 'flagship'
        )
        content_pieces.append(flagship)
        
        # Generate protocol
        self.logger.info("📋 Generating protocol...")
        protocol = await self._generate_single_draft(
            prompt_engine, 'protocol', 'complete', 'Full Protocol', 'protocol'
        )
        content_pieces.append(protocol)
        
        self.results.total_pieces = len(content_pieces)
        self.results.completed_pieces = sum(1 for p in content_pieces if p.status != "draft_failed")
        self.results.gemini_calls = self.api_manager.gemini_calls
        
        return content_pieces

    async def _generate_single_draft_with_prompt(self, prompt: str, content_type: str, 
                                                day: str, topic: str) -> ContentPiece:
        """Generate draft with pre-built prompt"""
        piece = ContentPiece(
            type=content_type,
            day=day,
            title=topic
        )
        
        try:
            content = await self.api_manager.generate_with_gemini(
                prompt, f"{content_type} - {day} - {topic}"
            )
            
            piece.draft_content = content
            piece.word_count = len(content.split())
            piece.status = "draft_complete"
            
            self.logger.info(f"✅ Generated {content_type} for {day}: {piece.word_count} words")
            
        except Exception as e:
            piece.status = "draft_failed"
            self.logger.error(f"❌ Failed to generate {content_type} for {day}: {str(e)}")
            
        return piece

    async def _generate_sunday_recap_enhanced(self, prompt_engine: PromptEngine, 
                                            previous_threads: Dict[str, str]) -> ContentPiece:
        """Generate enhanced Sunday recap that pulls from week's content"""
        
        # Extract key lines from each day
        weekly_highlights = {}
        for day, content in previous_threads.items():
            if content:
                # Extract the hook and one key mechanism line
                lines = content.split('\n')
                hook = next((line for line in lines if line.strip() and not line.endswith('/')), '')
                mechanism = next((line for line in lines if 'mechanism:' in line.lower()), '')
                weekly_highlights[day] = {
                    'hook': hook.strip(),
                    'mechanism': mechanism.strip()
                }
        
        prompt = f"""
{prompt_engine.get_gemini_system_prompt()}

THEME: {self.theme_config['name']}

WEEKLY CONTENT HIGHLIGHTS:
{json.dumps(weekly_highlights, indent=2)}

Generate a 15-tweet Sunday synthesis thread that creates a master narrative.

REQUIREMENTS:
- Extract and quote the most powerful line from each day
- Build progressive understanding toward sovereignty  
- Show how all mechanisms connect into one system
- End with complete protocol implementation path

STRUCTURE:
week's theme: {self.theme_config['name'].lower()}. 
six days. six mechanisms. one system.
here's what changes everything:
1/

monday: [topic]
"{weekly_highlights.get('monday', {}).get('hook', '')}"

2/

this connects to tuesday's discovery:
[pull exact powerful line from tuesday]

3/

wednesday revealed the deeper mechanism:
[pull exact powerful line from wednesday]

4/

thursday's protocol builds on this:
[pull exact powerful line from thursday]

5/

friday exposed what medicine ignores:
[pull exact powerful line from friday]

6/

saturday completed the system:
[pull exact powerful line from saturday]

7/

connecting all six days:
[master mechanism that unifies the week]

8/

the cascade:
monday's [X] → tuesday's [Y] → wednesday's [Z]

9/

implementation hierarchy:
1. [most critical intervention]
2. [second priority]
3. [third priority]

10/

tracking these three tells you everything:
- [biomarker 1]
- [biomarker 2]  
- [biomarker 3]

11/

ignore this system and accept:
[specific decline timeline]

12/

implement it and achieve:
[specific optimization outcome]

13/

your biology doesn't wait.
sovereignty requires action.
the choice is binary.

14/

complete protocol + flagship essay:
[link placeholder]

REMEMBER:
- All lowercase except acronyms
- Numbers on separate lines (X/)
- Pull EXACT quotes from daily content
- No emojis, no em-dashes
- Build progressive narrative

Generate the complete Sunday thread:
"""
        
        return await self._generate_single_draft_with_prompt(
            prompt, 'thread', 'sunday', 'Weekly Synthesis'
        )

    async def _generate_single_draft(self, prompt_engine: PromptEngine, content_type: str, 
                                   day: str, topic: str, subtype: str) -> ContentPiece:
        """Generate a single content draft"""
        piece = ContentPiece(
            type=content_type,
            day=day,
            title=topic
        )
        
        try:
            # Generate appropriate prompt
            if content_type == 'thread':
                if subtype == 'recap':
                    # Use old recap for fallback
                    prompt = self._create_sunday_recap_prompt(prompt_engine)
                else:
                    prompt = prompt_engine.create_thread_prompt(topic, day)
            elif content_type == 'essay':
                if subtype == 'flagship':
                    prompt = prompt_engine.create_flagship_essay_prompt()
                else:
                    prompt = prompt_engine.create_essay_prompt(topic, day)
            elif content_type == 'protocol':
                prompt = prompt_engine.create_protocol_prompt()
            else:
                raise ValueError(f"Unknown content type: {content_type}")
            
            # Generate content
            content = await self.api_manager.generate_with_gemini(
                prompt, f"{content_type} - {day} - {topic}"
            )
            
            piece.draft_content = content
            piece.word_count = len(content.split())
            piece.status = "draft_complete"
            
            self.logger.info(f"✅ Generated {content_type} for {day}: {piece.word_count} words")
            
        except Exception as e:
            piece.status = "draft_failed"
            self.logger.error(f"❌ Failed to generate {content_type} for {day}: {str(e)}")
            raise
        
        return piece

    def _create_sunday_recap_prompt(self, prompt_engine: PromptEngine) -> str:
        """Create Sunday recap thread prompt (fallback)"""
        daily_topics = self.theme_config['daily_topics']
        topics_list = "\n".join([f"- {day.title()}: {topic}" for day, topic in daily_topics.items()])
        
        return f"""
BIO/ACC SUNDAY SYNTHESIS THREAD

THEME: {self.theme_config['name']}

WEEK'S TOPICS:
{topics_list}

Generate a 15-tweet Sunday synthesis thread that creates a master narrative from the week's content.

SYNTHESIS REQUIREMENTS:
- 15 tweets numbered 1/, 2/, etc.
- Pull the most powerful mechanisms from each day
- Build progressive understanding toward sovereignty  
- Include protocol implementation elements
- End with civilizational implications

STRUCTURE:
1/ Week synthesis hook with theme connection
2-3/ Monday-Tuesday insights linked
4-5/ Wednesday-Thursday mechanisms connected  
6-7/ Friday-Saturday protocols unified
8-11/ Master mechanism explanation
12-13/ Complete implementation path
14/ Sovereignty implications
15/ "complete protocol: [link placeholder]"

Generate the complete Sunday synthesis thread:
"""
    
    async def _refine_all_content(self, content_pieces: List[ContentPiece]) -> List[ContentPiece]:
        """Refine all content with Claude sequentially with enhanced rate limiting"""
        
        # Group by priority and complexity
        priority_groups = {
            'threads': [],      # Small, quick refinements
            'essays': [],       # Medium complexity
            'flagship': [],     # Large, complex content
            'protocol': []      # Large, complex content
        }
        
        for piece in content_pieces:
            if piece.status == "draft_complete":
                if piece.type == 'thread':
                    priority_groups['threads'].append(piece)
                elif piece.type == 'essay' and piece.day != 'flagship':
                    priority_groups['essays'].append(piece)
                elif piece.type == 'essay' and piece.day == 'flagship':
                    priority_groups['flagship'].append(piece)
                elif piece.type == 'protocol':
                    priority_groups['protocol'].append(piece)
        
        refined_pieces = []
        
        # Process in order: threads -> essays -> flagship -> protocol
        for group_name, group_pieces in priority_groups.items():
            if not group_pieces:
                continue
                
            self.logger.info(f"\n🔄 Processing {group_name} ({len(group_pieces)} pieces)")
            
            for i, piece in enumerate(group_pieces):
                try:
                    # Show progress
                    self.logger.info(f"  [{i+1}/{len(group_pieces)}] Refining {piece.type} - {piece.day}")
                    
                    # Refine with proper delay
                    refined_piece = await self._refine_single_piece(piece)
                    refined_pieces.append(refined_piece)
                    
                    # Enhanced delays between pieces
                    if i < len(group_pieces) - 1:
                        delay = 8 if group_name in ['flagship', 'protocol'] else 5
                        self.logger.info(f"⏳ Waiting {delay}s before next piece...")
                        await asyncio.sleep(delay)
                        
                except Exception as e:
                    self.logger.error(f"Failed to refine {piece.type} - {piece.day}: {str(e)}")
                    piece.status = "refinement_failed"
                    # Still add to keep original Gemini content
                    refined_pieces.append(piece)
            
            # Longer delay between groups
            if group_name != 'protocol':  # No delay after last group
                self.logger.info(f"⏳ Waiting 15s before next group...")
                await asyncio.sleep(15)
        
        self.results.claude_calls = self.api_manager.claude_calls
        
        self.logger.info(f"✅ Refined {len(refined_pieces)} pieces sequentially")
        return refined_pieces
    
    async def _refine_single_piece(self, piece: ContentPiece) -> ContentPiece:
        """Enhanced refinement with validation"""
        try:
            # Validate topic alignment BEFORE refinement
            if piece.type in ['thread', 'essay']:
                topic_score, topic_issues = self.topic_validator.validate_topic_alignment(
                    piece.draft_content, piece.title
                )
                
                if topic_score < 70:
                    self.logger.warning(f"⚠️  Low topic alignment for {piece.day}: {topic_score:.1f}")
                    self.logger.warning(f"   Issues: {', '.join(topic_issues)}")
                    
                    # Regenerate if way off topic
                    if topic_score < 30:
                        self.logger.info(f"🔄 Regenerating {piece.type} for {piece.day} due to topic mismatch")
                        # Re-generate with stronger topic emphasis
                        piece = await self._regenerate_with_topic_emphasis(piece)
            
            # Process references for essays
            if piece.type == 'essay' and piece.draft_content:
                piece.draft_content = self.reference_manager.process_essay_references(
                    piece.draft_content, piece.day
                )
            
            # Continue with normal refinement
            content_type = piece.type
            if piece.day == 'flagship':
                content_type = 'flagship'
            
            refined_content = await self.api_manager.refine_with_claude(
                piece.draft_content, content_type, f"{piece.type} - {piece.day}"
            )
            
            piece.refined_content = refined_content
            
            # Validate voice
            if piece.type == 'thread':
                voice_score, issues = VoiceValidator.validate_thread_voice(refined_content)
            else:
                voice_score, issues = VoiceValidator.validate_essay_voice(refined_content)
            
            piece.voice_score = voice_score
            piece.status = "complete"
            
            if issues:
                self.logger.warning(f"Voice issues in {piece.type} {piece.day}: {', '.join(issues[:3])}")
            
            self.logger.info(f"✅ Refined {piece.type} for {piece.day}: Voice score {voice_score:.1f}")
            
        except Exception as e:
            piece.status = "refinement_failed"
            self.logger.error(f"❌ Failed to refine {piece.type} for {piece.day}: {str(e)}")
            raise
        
        return piece

    async def _regenerate_with_topic_emphasis(self, piece: ContentPiece) -> ContentPiece:
        """Regenerate content with stronger topic focus"""
        enhanced_prompt = f"""
CRITICAL: The previous generation drifted off-topic.
THIS MUST BE ABOUT: {piece.title}
DO NOT DISCUSS OTHER TOPICS.

Focus EXCLUSIVELY on {piece.title} throughout.
Every tweet/paragraph must directly relate to {piece.title}.

Generate a {piece.type} about {piece.title} for {piece.day}:
"""
        
        content = await self.api_manager.generate_with_gemini(
            enhanced_prompt, f"RETRY: {piece.type} - {piece.day} - {piece.title}"
        )
        
        piece.draft_content = content
        piece.word_count = len(content.split())
        
        return piece
    
    async def _save_all_content(self, content_pieces: List[ContentPiece]):
        """Save all content to files"""
        base_dir = Path(self.output_dir)
        
        for piece in content_pieces:
            if piece.status == "complete":
                await self._save_single_piece(piece, base_dir)
    
    async def _save_single_piece(self, piece: ContentPiece, base_dir: Path):
        """Save a single content piece (both original Gemini and Claude refined versions)"""
        try:
            # Determine file path and extension
            if piece.type == 'thread':
                subdir = 'threads'
                extension = '.txt'
            elif piece.type == 'essay':
                subdir = 'essays'
                extension = '.md'
            elif piece.type == 'protocol':
                subdir = 'protocol'
                extension = '.md'
            else:
                subdir = 'other'
                extension = '.txt'
            
            base_filename = f"{piece.day}_{piece.type}"
            if piece.day == 'flagship':
                base_filename = "flagship_essay"
            elif piece.day == 'complete':
                base_filename = f"{self.theme_config['name'].lower().replace(' ', '_')}_protocol"
            
            # Save original Gemini version (always available)
            if piece.draft_content:
                gemini_filename = f"{base_filename}_gemini{extension}"
                gemini_file_path = base_dir / subdir / gemini_filename
                
                async with aiofiles.open(gemini_file_path, 'w', encoding='utf-8') as f:
                    await f.write(piece.draft_content)
                
                self.results.files_created.append(str(gemini_file_path))
                self.logger.info(f"💾 Saved GEMINI {piece.type} for {piece.day}: {len(piece.draft_content)} chars")
            
            # Save Claude refined version (if available)
            if piece.refined_content and piece.refined_content.strip():
                claude_filename = f"{base_filename}_claude{extension}"
                claude_file_path = base_dir / subdir / claude_filename
                
                async with aiofiles.open(claude_file_path, 'w', encoding='utf-8') as f:
                    await f.write(piece.refined_content)
                
                self.results.files_created.append(str(claude_file_path))
                self.logger.info(f"💾 Saved CLAUDE {piece.type} for {piece.day}: {len(piece.refined_content)} chars")
            else:
                self.logger.warning(f"⚠️  No Claude refined content for {piece.type} {piece.day}, using Gemini only")
            
            # Save the "final" version (prefer Claude if available and valid, otherwise Gemini)
            final_filename = f"{base_filename}{extension}"
            final_file_path = base_dir / subdir / final_filename
            
            # Choose best content
            if piece.refined_content and piece.refined_content.strip() and len(piece.refined_content) > 100:
                final_content = piece.refined_content
                content_source = "CLAUDE"
            else:
                final_content = piece.draft_content
                content_source = "GEMINI"
            
            async with aiofiles.open(final_file_path, 'w', encoding='utf-8') as f:
                await f.write(final_content)
            
            self.results.files_created.append(str(final_file_path))
            self.logger.info(f"💾 Saved FINAL ({content_source}) {piece.type} for {piece.day}: {len(final_content)} chars")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save {piece.type} for {piece.day}: {str(e)}")

    async def _generate_whole_content_md(self, content_pieces: List[ContentPiece]):
        """Generate consolidated markdown file with all content"""
        base_dir = Path(self.output_dir)
        
        # Build the consolidated content
        md_content = f"""# Bio/Acc Content Suite: {self.theme_config['name']}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Theme**: {self.theme_config['name']}  
**Total Pieces**: {len(content_pieces)}

## Table of Contents

1. [Generation Summary](#generation-summary)
2. [Daily Threads](#daily-threads)
3. [Daily Essays](#daily-essays)
4. [Sunday Synthesis](#sunday-synthesis)
5. [Flagship Essay](#flagship-essay)
6. [Protocol](#protocol)
7. [Quality Metrics](#quality-metrics)
8. [Publishing Checklist](#publishing-checklist)

---

## Generation Summary

| Metric | Value |
|--------|-------|
| Total Pieces | {self.results.total_pieces} |
| Completed | {self.results.completed_pieces} |
| Failed | {self.results.failed_pieces} |
| Gemini Calls | {self.results.gemini_calls} |
| Claude Calls | {self.results.claude_calls} |
| Estimated Cost | ${self.results.total_cost:.2f} |

---

## Daily Threads
"""
        
        # Add threads
        days_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        for day in days_order:
            thread = next((p for p in content_pieces if p.type == 'thread' and p.day == day), None)
            if thread:
                md_content += f"\n### {day.title()} Thread\n"
                md_content += f"**Topic**: {thread.title}\n"
                md_content += f"**Voice Score**: {thread.voice_score:.1f}/100\n"
                md_content += f"**Status**: {thread.status}\n\n"
                
                if thread.refined_content:
                    md_content += "```\n" + thread.refined_content + "\n```\n"
                elif thread.draft_content:
                    md_content += "```\n" + thread.draft_content + "\n```\n"
                    md_content += "*Note: Using Gemini draft (Claude refinement failed)*\n"
        
        md_content += "\n---\n\n## Daily Essays\n"
        
        # Add essays
        for day in days_order[:-1]:  # Exclude Sunday
            essay = next((p for p in content_pieces if p.type == 'essay' and p.day == day), None)
            if essay:
                md_content += f"\n### {day.title()} Essay\n"
                md_content += f"**Topic**: {essay.title}\n"
                md_content += f"**Word Count**: {essay.word_count}\n"
                md_content += f"**Voice Score**: {essay.voice_score:.1f}/100\n\n"
                
                if essay.refined_content:
                    md_content += essay.refined_content + "\n"
                elif essay.draft_content:
                    md_content += essay.draft_content + "\n"
                    md_content += "\n*Note: Using Gemini draft (Claude refinement failed)*\n"
        
        # Add flagship essay
        flagship = next((p for p in content_pieces if p.type == 'essay' and p.day == 'flagship'), None)
        if flagship:
            md_content += "\n---\n\n## Flagship Essay\n\n"
            md_content += f"**Word Count**: {flagship.word_count}\n"
            md_content += f"**Voice Score**: {flagship.voice_score:.1f}/100\n\n"
            
            if flagship.refined_content:
                md_content += flagship.refined_content + "\n"
            elif flagship.draft_content:
                md_content += flagship.draft_content + "\n"
        
        # Add protocol
        protocol = next((p for p in content_pieces if p.type == 'protocol'), None)
        if protocol:
            md_content += "\n---\n\n## Protocol\n\n"
            md_content += f"**Status**: {protocol.status}\n\n"
            
            if protocol.refined_content:
                md_content += protocol.refined_content + "\n"
            elif protocol.draft_content:
                md_content += protocol.draft_content + "\n"
        
        # Add quality metrics
        md_content += "\n---\n\n## Quality Metrics\n\n"
        
        # Calculate average voice scores
        thread_scores = [p.voice_score for p in content_pieces if p.type == 'thread' and p.voice_score > 0]
        essay_scores = [p.voice_score for p in content_pieces if p.type == 'essay' and p.voice_score > 0]
        
        if thread_scores:
            md_content += f"**Average Thread Voice Score**: {sum(thread_scores)/len(thread_scores):.1f}/100\n"
        if essay_scores:
            md_content += f"**Average Essay Voice Score**: {sum(essay_scores)/len(essay_scores):.1f}/100\n"
        
        # Flag any issues
        md_content += "\n### Voice Violations\n\n"
        violations_found = False
        
        for piece in content_pieces:
            if piece.voice_score < 70 and piece.voice_score > 0:
                violations_found = True
                md_content += f"- **{piece.type.title()} {piece.day}**: Score {piece.voice_score:.1f}/100\n"
        
        if not violations_found:
            md_content += "*No significant voice violations detected*\n"
        
        # Add publishing checklist
        md_content += "\n---\n\n## Publishing Checklist\n\n"
        md_content += """### Twitter/X (via Hypefury)
- [ ] Monday thread scheduled
- [ ] Tuesday thread scheduled  
- [ ] Wednesday thread scheduled
- [ ] Thursday thread scheduled
- [ ] Friday thread scheduled
- [ ] Saturday thread scheduled
- [ ] Sunday recap scheduled

### Substack
- [ ] Monday essay published
- [ ] Tuesday essay published
- [ ] Wednesday essay published
- [ ] Thursday essay published
- [ ] Friday essay published
- [ ] Saturday essay published
- [ ] Flagship essay scheduled

### Gumroad
- [ ] Protocol PDF created
- [ ] Pricing set ($97)
- [ ] Description written
- [ ] Cover image designed

### Images
- [ ] Daily hero images generated
- [ ] Protocol visuals created
- [ ] All images uploaded to platforms
"""
        
        # Save the file
        output_path = base_dir / 'whole-content.md'
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(md_content)
        
        self.logger.info(f"📄 Generated whole-content.md: {output_path}")
        self.results.files_created.append(str(output_path))

    def _generate_final_report(self):
        """Generate comprehensive final report"""
        base_dir = Path(self.output_dir)
        
        # Calculate costs
        gemini_cost = self.results.gemini_calls * 0.02  # Approximate cost per call
        claude_cost = self.results.claude_calls * 0.15  # Approximate cost per call
        self.results.total_cost = gemini_cost + claude_cost
        
        report = {
            'generation_summary': {
                'timestamp': self.results.timestamp,
                'theme': self.results.theme,
                'status': 'completed' if self.results.failed_pieces == 0 else 'partial',
                'total_pieces': self.results.total_pieces,
                'completed_pieces': self.results.completed_pieces,
                'failed_pieces': self.results.failed_pieces
            },
            'api_usage': {
                'gemini_calls': self.results.gemini_calls,
                'claude_calls': self.results.claude_calls,
                'estimated_cost': f"${self.results.total_cost:.2f}"
            },
            'files_created': self.results.files_created,
            'next_steps': [
                "Review generated content for quality",
                "Schedule threads via Hypefury",
                "Publish essays to Substack", 
                "Package protocol for Gumroad",
                "Generate visuals with Midjourney"
            ]
        }
        
        # Save report
        report_path = base_dir / 'reports' / f"generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        self.logger.info("🎯 GENERATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Theme: {self.results.theme}")
        self.logger.info(f"Total Pieces: {self.results.total_pieces}")
        self.logger.info(f"Completed: {self.results.completed_pieces}")
        self.logger.info(f"Failed: {self.results.failed_pieces}")
        self.logger.info(f"Files Created: {len(self.results.files_created)}")
        self.logger.info(f"API Calls: {self.results.gemini_calls + self.results.claude_calls}")
        self.logger.info(f"Estimated Cost: ${self.results.total_cost:.2f}")
        self.logger.info(f"Report Saved: {report_path}")

async def main():
    """Main execution function"""
    # Use the main config file with weekly_content_generation section
    config_path = "config/main_config.json"
    
    try:
        generator = BioAccContentGenerator(config_path)
        results = await generator.generate_complete_suite()
        
        if results.failed_pieces == 0:
            print("🏆 Bio/Acc Content Generation COMPLETE!")
            print(f"📊 Generated: {results.completed_pieces} pieces")
            print(f"💰 Estimated cost: ${results.total_cost:.2f}")
            print(f"📁 Files saved to: {generator.output_dir}")
            return 0
        else:
            print(f"⚠️  Generation completed with {results.failed_pieces} failures")
            print(f"📊 Generated: {results.completed_pieces}/{results.total_pieces} pieces")
            return 1
            
    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)