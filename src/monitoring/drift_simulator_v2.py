"""
Enhanced Drift Simulator - Creates dramatic, realistic drift scenarios
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re
from datetime import datetime, timedelta


class EnhancedDriftSimulator:
    """
    Simulates various types of data drift with dramatic, visible changes
    """
    
    def __init__(self):
        self.drift_scenarios = {}
    
    # ==========================================
    # SCENARIO 1: EXTREME TEXT LENGTH CHANGE
    # ==========================================
    
    def simulate_extreme_length_drift(self, 
                                      data: pd.DataFrame, 
                                      factor: float = 3.0,
                                      proportion: float = 1.0) -> pd.DataFrame:
        """
        Simulate extreme text length changes
        
        Factor 3.0 = Triple the length (dramatic!)
        Proportion 1.0 = Apply to all samples
        
        Real-world: Users start writing much longer descriptions
        """
        drifted = data.copy()
        n_samples = int(len(drifted) * proportion)
        indices = np.random.choice(drifted.index, n_samples, replace=False)
        
        if factor > 1:
            # Make text longer by repeating with variations
            for idx in indices:
                original_text = drifted.loc[idx, 'text']
                
                # Add different variations
                variations = [
                    original_text,
                    original_text.replace('.', '. Also,'),
                    original_text.replace('I ', 'I really '),
                    f"{original_text} This has been going on for a while.",
                ]
                
                # Repeat until we reach desired length
                repeated_text = original_text
                while len(repeated_text) < len(original_text) * factor:
                    repeated_text += " " + np.random.choice(variations)
                
                drifted.loc[idx, 'text'] = repeated_text[:int(len(original_text) * factor)]
        
        else:
            # Make text shorter (truncate)
            for idx in indices:
                original_text = drifted.loc[idx, 'text']
                new_length = int(len(original_text) * factor)
                drifted.loc[idx, 'text'] = original_text[:new_length]
        
        print(f"✅ Length Drift: Changed {n_samples} samples by {factor}x")
        return drifted
    
    # ==========================================
    # SCENARIO 2: HEAVY SLANG INJECTION
    # ==========================================
    
    def simulate_heavy_slang(self, 
                            data: pd.DataFrame,
                            intensity: float = 0.8) -> pd.DataFrame:
        """
        Replace formal language with heavy Gen Z slang
        
        Intensity 0.8 = Replace 80% of matches
        
        Real-world: Platform becomes popular with younger demographic
        """
        drifted = data.copy()
        
        # Comprehensive slang dictionary
        slang_dict = {
            # Mental health terms
            'depressed': 'hella down bad',
            'depression': 'big sad',
            'sad': 'down bad',
            'very sad': 'mad depresso',
            'anxious': 'lowkey stressed',
            'anxiety': 'major anxiety vibes',
            'stressed': 'stressed af',
            'stress': 'stress',
            'worried': 'lowkey worried',
            'scared': 'shook',
            'afraid': 'deadass scared',
            'hopeless': 'no cap hopeless',
            'lonely': 'feeling lonely fr',
            'alone': 'mad lonely',
            
            # Intensity modifiers
            'very': 'hella',
            'really': 'deadass',
            'extremely': 'mad',
            'quite': 'lowkey',
            'somewhat': 'kinda',
            
            # Common phrases
            'I feel': 'im feeling',
            'I am': 'im',
            'I have': 'ive got',
            'cannot': 'cant',
            'do not': 'dont',
            
            # Emotional states
            'happy': 'vibing',
            'good': 'gucci',
            'bad': 'trash',
            'terrible': 'absolute trash',
            'horrible': 'mad trash',
            'awful': 'straight up awful',
            
            # Social terms
            'friends': 'squad',
            'friend': 'homie',
            'people': 'folks',
            'person': 'dude',
            
            # Time terms
            'always': 'fr always',
            'never': 'lowkey never',
            'sometimes': 'sometimes ngl',
            
            # Sentence endings (add randomly)
            '.': np.random.choice(['.', ' fr.', ' ngl.', ' no cap.', ' tbh.'], p=[0.3, 0.2, 0.2, 0.2, 0.1])
        }
        
        # Apply slang transformations
        for idx in drifted.index:
            if np.random.random() < intensity:
                text = drifted.loc[idx, 'text']
                
                # Apply each substitution
                for formal, slang in slang_dict.items():
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(formal), re.IGNORECASE)
                    if np.random.random() < intensity:
                        text = pattern.sub(slang, text)
                
                # Add internet-style punctuation
                if np.random.random() < 0.5:
                    text = text.replace('!', '!!!')
                
                # Random caps for emphasis
                if np.random.random() < 0.3:
                    words = text.split()
                    if len(words) > 3:
                        emphasis_idx = np.random.randint(0, len(words))
                        words[emphasis_idx] = words[emphasis_idx].upper()
                        text = ' '.join(words)
                
                drifted.loc[idx, 'text'] = text
        
        affected = int(len(drifted) * intensity)
        print(f"✅ Slang Drift: Transformed {affected} samples with Gen Z slang")
        return drifted
    
    # ==========================================
    # SCENARIO 3: DRAMATIC CLASS IMBALANCE
    # ==========================================
    
    def simulate_population_collapse(self,
                                     data: pd.DataFrame,
                                     dominant_class: str = 'Anxiety',
                                     dominance: float = 0.8) -> pd.DataFrame:
        """
        Simulate one class becoming dominant (80%+ of data)
        
        Real-world: Platform becomes known for one issue (e.g., anxiety support)
        Everyone with that issue joins, others leave
        """
        drifted = data.copy()
        
        # Get samples of dominant class
        dominant_samples = drifted[drifted['label'] == dominant_class]
        other_samples = drifted[drifted['label'] != dominant_class]
        
        # Calculate target distribution
        target_dominant = int(len(drifted) * dominance)
        target_other = len(drifted) - target_dominant
        
        # Oversample dominant class
        if len(dominant_samples) < target_dominant:
            dominant_resampled = dominant_samples.sample(
                n=target_dominant, 
                replace=True, 
                random_state=42
            )
        else:
            dominant_resampled = dominant_samples.sample(
                n=target_dominant, 
                replace=False, 
                random_state=42
            )
        
        # Undersample other classes
        other_resampled = other_samples.sample(
            n=target_other, 
            replace=False if len(other_samples) >= target_other else True,
            random_state=42
        )
        
        # Combine
        drifted = pd.concat([dominant_resampled, other_resampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Population Collapse: {dominant_class} now {dominance*100}% of data")
        print(f"   Before: {(data['label'] == dominant_class).mean()*100:.1f}%")
        print(f"   After:  {(drifted['label'] == dominant_class).mean()*100:.1f}%")
        
        return drifted
    
    # ==========================================
    # SCENARIO 4: SENTIMENT FLIP
    # ==========================================
    
    def simulate_sentiment_flip(self,
                                data: pd.DataFrame,
                                proportion: float = 0.6) -> pd.DataFrame:
        """
        Flip sentiment words (positive ↔ negative)
        
        Real-world: Sarcasm, irony, or linguistic drift
        """
        drifted = data.copy()
        
        # Sentiment flip dictionary
        flips = {
            'good': 'bad',
            'bad': 'good',
            'great': 'terrible',
            'terrible': 'great',
            'happy': 'sad',
            'sad': 'happy',
            'love': 'hate',
            'hate': 'love',
            'better': 'worse',
            'worse': 'better',
            'hope': 'despair',
            'hopeful': 'hopeless',
            'hopeless': 'hopeful',
            'positive': 'negative',
            'negative': 'positive',
            'calm': 'anxious',
            'peaceful': 'chaotic',
            'stable': 'unstable',
            'well': 'unwell',
        }
        
        n_samples = int(len(drifted) * proportion)
        indices = np.random.choice(drifted.index, n_samples, replace=False)
        
        for idx in indices:
            text = drifted.loc[idx, 'text']
            
            for original, flipped in flips.items():
                # Case-insensitive replacement
                pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
                text = pattern.sub(flipped, text)
            
            drifted.loc[idx, 'text'] = text
        
        print(f"✅ Sentiment Flip: Flipped sentiment in {n_samples} samples")
        return drifted
    
    # ==========================================
    # SCENARIO 5: FORMALITY SHIFT
    # ==========================================
    
    def simulate_formality_shift(self,
                                 data: pd.DataFrame,
                                 direction: str = 'informal',
                                 intensity: float = 0.7) -> pd.DataFrame:
        """
        Shift between formal/informal language
        
        Real-world: Platform changes (clinical → casual, or vice versa)
        """
        drifted = data.copy()
        
        if direction == 'informal':
            # Formal → Informal
            transforms = {
                'I am experiencing': 'Im feeling',
                'I have been feeling': 'Ive been feeling',
                'experiencing': 'going through',
                'significant': 'major',
                'difficulty': 'trouble',
                'unable to': 'cant',
                'attempt to': 'try to',
                'regarding': 'about',
                'concerning': 'worrying',
                'substantial': 'big',
                'frequently': 'a lot',
                'occasionally': 'sometimes',
                'extremely': 'super',
                'particularly': 'especially',
                'currently': 'right now',
                'recently': 'lately',
                'therapist': 'counselor',
                'medication': 'meds',
                'symptoms': 'issues',
            }
        else:
            # Informal → Formal
            transforms = {
                'Im': 'I am',
                'Ive': 'I have',
                'cant': 'cannot',
                'dont': 'do not',
                'wont': 'will not',
                'feeling': 'experiencing',
                'super': 'extremely',
                'a lot': 'frequently',
                'kinda': 'somewhat',
                'really': 'significantly',
                'big': 'substantial',
                'meds': 'medication',
                'issues': 'symptoms',
            }
        
        n_samples = int(len(drifted) * intensity)
        indices = np.random.choice(drifted.index, n_samples, replace=False)
        
        for idx in indices:
            text = drifted.loc[idx, 'text']
            
            for original, transformed in transforms.items():
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                text = pattern.sub(transformed, text)
            
            drifted.loc[idx, 'text'] = text
        
        print(f"✅ Formality Shift: Transformed {n_samples} to {direction}")
        return drifted
    
    # ==========================================
    # SCENARIO 6: NOISE INJECTION
    # ==========================================
    
    def simulate_heavy_noise(self,
                            data: pd.DataFrame,
                            noise_level: float = 0.3) -> pd.DataFrame:
        """
        Add heavy typos, repeated chars, emojis
        
        Real-world: Mobile users, emotional typing, lower quality input
        """
        drifted = data.copy()
        
        emojis = ['😭', '😔', '😢', '😰', '😨', '😱', '💔', '😞', '😟', '😥']
        
        for idx in drifted.index:
            if np.random.random() < noise_level:
                text = drifted.loc[idx, 'text']
                words = text.split()
                
                # Add typos (swap adjacent letters)
                if len(words) > 0 and np.random.random() < 0.5:
                    word_idx = np.random.randint(0, len(words))
                    word = words[word_idx]
                    if len(word) > 3:
                        pos = np.random.randint(0, len(word)-1)
                        word_list = list(word)
                        word_list[pos], word_list[pos+1] = word_list[pos+1], word_list[pos]
                        words[word_idx] = ''.join(word_list)
                
                # Repeat characters for emphasis
                if len(words) > 0 and np.random.random() < 0.4:
                    word_idx = np.random.randint(0, len(words))
                    word = words[word_idx]
                    if len(word) > 2:
                        char_idx = np.random.randint(0, len(word))
                        words[word_idx] = word[:char_idx+1] + word[char_idx]*2 + word[char_idx+1:]
                
                # Add emojis
                if np.random.random() < 0.5:
                    emoji = np.random.choice(emojis)
                    words.append(emoji)
                
                # Multiple punctuation
                text = ' '.join(words)
                if np.random.random() < 0.4:
                    text = text.replace('.', '...')
                    text = text.replace('!', '!!!')
                
                # Random caps
                if np.random.random() < 0.3:
                    text = text.upper()
                
                drifted.loc[idx, 'text'] = text
        
        affected = int(len(drifted) * noise_level)
        print(f"✅ Noise Injection: Added heavy noise to {affected} samples")
        return drifted
    
    # ==========================================
    # COMBINED SCENARIOS
    # ==========================================
    
    def simulate_catastrophic_drift(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine multiple drift types for catastrophic failure
        
        Real-world: Multiple factors change simultaneously
        """
        print("\n🚨 SIMULATING CATASTROPHIC DRIFT...")
        print("=" * 60)
        
        drifted = data.copy()
        
        # Apply multiple drifts in sequence
        drifted = self.simulate_extreme_length_drift(drifted, factor=0.4, proportion=0.6)  # 60% shorter
        drifted = self.simulate_heavy_slang(drifted, intensity=0.9)  # Heavy slang
        drifted = self.simulate_heavy_noise(drifted, noise_level=0.5)  # Lots of noise
        drifted = self.simulate_population_collapse(drifted, dominant_class='Anxiety', dominance=0.85)
        
        print("=" * 60)
        print("🚨 CATASTROPHIC DRIFT COMPLETE!")
        return drifted
    
    # ==========================================
    # COMPARISON REPORT
    # ==========================================
    
    def generate_comparison_report(self, 
                                   original: pd.DataFrame, 
                                   drifted: pd.DataFrame) -> Dict:
        """
        Generate detailed comparison between original and drifted data
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(drifted),
        }
        
        # Text length statistics
        orig_lengths = original['text'].str.len()
        drift_lengths = drifted['text'].str.len()
        
        report['text_length'] = {
            'original_mean': float(orig_lengths.mean()),
            'drifted_mean': float(drift_lengths.mean()),
            'change_pct': float(((drift_lengths.mean() - orig_lengths.mean()) / orig_lengths.mean()) * 100),
            'original_std': float(orig_lengths.std()),
            'drifted_std': float(drift_lengths.std()),
        }
        
        # Word count statistics
        orig_words = original['text'].str.split().str.len()
        drift_words = drifted['text'].str.split().str.len()
        
        report['word_count'] = {
            'original_mean': float(orig_words.mean()),
            'drifted_mean': float(drift_words.mean()),
            'change_pct': float(((drift_words.mean() - orig_words.mean()) / orig_words.mean()) * 100),
        }
        
        # Class distribution (if label exists)
        if 'label' in original.columns and 'label' in drifted.columns:
            orig_dist = original['label'].value_counts(normalize=True).to_dict()
            drift_dist = drifted['label'].value_counts(normalize=True).to_dict()
            
            report['class_distribution'] = {
                'original': {k: float(v) for k, v in orig_dist.items()},
                'drifted': {k: float(v) for k, v in drift_dist.items()},
            }
        
        # Sample texts for inspection
        report['sample_comparisons'] = []
        for i in range(min(3, len(drifted))):
            report['sample_comparisons'].append({
                'original': original.iloc[i]['text'][:200],
                'drifted': drifted.iloc[i]['text'][:200],
            })
        
        return report
    
    def print_report(self, report: Dict):
        """Pretty print comparison report"""
        print("\n" + "="*80)
        print("📊 DRIFT COMPARISON REPORT")
        print("="*80)
        
        print(f"\n📅 Timestamp: {report['timestamp']}")
        print(f"📦 Samples: {report['n_samples']}")
        
        print(f"\n📏 TEXT LENGTH:")
        tl = report['text_length']
        print(f"   Original: {tl['original_mean']:.1f} ± {tl['original_std']:.1f} chars")
        print(f"   Drifted:  {tl['drifted_mean']:.1f} ± {tl['drifted_std']:.1f} chars")
        print(f"   Change:   {tl['change_pct']:+.1f}% {'🚨' if abs(tl['change_pct']) > 20 else '✅'}")
        
        print(f"\n📝 WORD COUNT:")
        wc = report['word_count']
        print(f"   Original: {wc['original_mean']:.1f} words")
        print(f"   Drifted:  {wc['drifted_mean']:.1f} words")
        print(f"   Change:   {wc['change_pct']:+.1f}% {'🚨' if abs(wc['change_pct']) > 20 else '✅'}")
        
        if 'class_distribution' in report:
            print(f"\n📊 CLASS DISTRIBUTION:")
            print("   Original:")
            for cls, pct in sorted(report['class_distribution']['original'].items()):
                print(f"      {cls:15s}: {pct*100:5.1f}%")
            print("   Drifted:")
            for cls, pct in sorted(report['class_distribution']['drifted'].items()):
                orig_pct = report['class_distribution']['original'].get(cls, 0)
                change = ((pct - orig_pct) / orig_pct * 100) if orig_pct > 0 else 0
                indicator = '🚨' if abs(change) > 50 else ('⚠️' if abs(change) > 20 else '✅')
                print(f"      {cls:15s}: {pct*100:5.1f}% ({change:+.0f}%) {indicator}")
        
        print(f"\n📄 SAMPLE COMPARISONS:")
        for i, sample in enumerate(report['sample_comparisons'], 1):
            print(f"\n   Sample {i}:")
            print(f"   Original: {sample['original']}...")
            print(f"   Drifted:  {sample['drifted']}...")
        
        print("\n" + "="*80)