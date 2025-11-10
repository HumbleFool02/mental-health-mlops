"""
Drift Simulator
Creates synthetic drift scenarios for testing drift detection
"""

import random

import numpy as np
import pandas as pd


class DriftSimulator:
    """
    Simulates various types of drift for testing
    """

    def __init__(self, seed: int = 42):
        """Initialize simulator with random seed"""
        np.random.seed(seed)
        random.seed(seed)

    def simulate_seasonal_drift(
        self, data: pd.DataFrame, season: str = "winter"
    ) -> pd.DataFrame:
        """
        Simulate seasonal language changes

        Args:
            data: Original dataset
            season: 'winter', 'summer', 'holiday'
        """
        modified_data = data.copy()

        seasonal_terms = {
            "winter": [
                ("sad", "seasonal affective disorder"),
                ("tired", "exhausted and cold"),
                ("alone", "isolated in winter"),
                ("dark", "long dark nights"),
            ],
            "summer": [
                ("anxious", "restless and hot"),
                ("worried", "concerned about summer"),
                ("stressed", "overwhelmed by heat"),
            ],
            "holiday": [
                ("depressed", "holiday depression"),
                ("lonely", "alone during holidays"),
                ("anxious", "holiday anxiety"),
                ("stressed", "holiday stress"),
            ],
        }

        terms = seasonal_terms.get(season, seasonal_terms["winter"])

        # Apply term substitutions to 30% of samples
        n_modify = int(0.3 * len(modified_data))
        indices = np.random.choice(len(modified_data), n_modify, replace=False)

        for idx in indices:
            text = modified_data.loc[idx, "text"]
            for old_term, new_term in terms:
                if old_term in text.lower():
                    text = text.replace(old_term, new_term)
                    text = text.replace(old_term.capitalize(), new_term.capitalize())
            modified_data.loc[idx, "text"] = text

        return modified_data

    def simulate_slang_emergence(
        self, data: pd.DataFrame, intensity: float = 0.2
    ) -> pd.DataFrame:
        """
        Simulate new slang/terminology emerging

        Args:
            data: Original dataset
            intensity: Proportion of samples to modify (0-1)
        """
        modified_data = data.copy()

        # Modern slang terms
        slang_substitutions = [
            ("very sad", "big sad"),
            ("sad", "down bad"),
            ("anxious", "stressed af"),
            ("depressed", "feeling down"),
            ("worried", "lowkey worried"),
            ("scared", "freaking out"),
            ("happy", "vibing"),
            ("feel bad", "not ok rn"),
        ]

        n_modify = int(intensity * len(modified_data))
        indices = np.random.choice(len(modified_data), n_modify, replace=False)

        for idx in indices:
            text = modified_data.loc[idx, "text"]
            # Apply random slang substitution
            old_term, new_term = random.choice(slang_substitutions)
            if old_term in text.lower():
                text = text.replace(old_term, new_term)
            modified_data.loc[idx, "text"] = text

        return modified_data

    def simulate_length_drift(
        self, data: pd.DataFrame, change_factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Simulate change in text length

        Args:
            data: Original dataset
            change_factor: Multiply length by this factor (>1 = longer, <1 = shorter)
        """
        modified_data = data.copy()

        if change_factor > 1:
            # Make texts longer by repeating
            modified_data["text"] = modified_data["text"].apply(
                lambda x: (x + " ") * int(change_factor)
            )
        else:
            # Make texts shorter by truncating
            modified_data["text"] = modified_data["text"].apply(
                lambda x: " ".join(x.split()[: int(len(x.split()) * change_factor)])
            )

        return modified_data

    def simulate_class_shift(
        self,
        data: pd.DataFrame,
        shift_from: str,
        shift_to: str,
        proportion: float = 0.3,
    ) -> pd.DataFrame:
        """
        Simulate population shift (more of one class, less of another)

        Args:
            data: Original dataset
            shift_from: Class to reduce
            shift_to: Class to increase
            proportion: Proportion of samples to shift
        """
        modified_data = data.copy()

        # Get samples from shift_from class
        from_samples = modified_data[modified_data["label"] == shift_from]
        n_shift = int(len(from_samples) * proportion)

        if n_shift > 0:
            # Get samples to shift
            shift_indices = from_samples.sample(n=n_shift, random_state=42).index

            # Get replacement samples from shift_to class
            to_samples = modified_data[modified_data["label"] == shift_to]
            replacement_samples = to_samples.sample(
                n=n_shift, replace=True, random_state=42
            )

            # Replace
            modified_data.loc[shift_indices, "text"] = replacement_samples[
                "text"
            ].values
            modified_data.loc[shift_indices, "label"] = shift_to

        return modified_data

    def simulate_noise_injection(
        self, data: pd.DataFrame, noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Add noise to text (typos, special characters)

        Args:
            data: Original dataset
            noise_level: Proportion of words to add noise to
        """
        modified_data = data.copy()

        def add_noise_to_text(text):
            words = text.split()
            n_noise = int(len(words) * noise_level)
            noise_indices = np.random.choice(
                len(words), min(n_noise, len(words)), replace=False
            )

            for idx in noise_indices:
                word = words[idx]
                # Random noise types
                noise_type = random.choice(["typo", "repeat", "special_char"])

                if noise_type == "typo" and len(word) > 3:
                    # Swap two characters
                    pos = random.randint(0, len(word) - 2)
                    word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
                elif noise_type == "repeat":
                    # Repeat character
                    pos = random.randint(0, len(word) - 1)
                    word = word[:pos] + word[pos] * 2 + word[pos + 1 :]
                elif noise_type == "special_char":
                    # Add special character
                    word = word + random.choice(["!", "?", "...", "!!"])

                words[idx] = word

            return " ".join(words)

        modified_data["text"] = modified_data["text"].apply(add_noise_to_text)
        return modified_data

    def simulate_gradual_drift(
        self, data: pd.DataFrame, drift_type: str = "seasonal", n_steps: int = 5
    ) -> list:
        """
        Simulate gradual drift over time

        Args:
            data: Original dataset
            drift_type: Type of drift to simulate
            n_steps: Number of time steps

        Returns:
            List of DataFrames showing progressive drift
        """
        drift_steps = [data.copy()]  # Original

        for step in range(1, n_steps + 1):
            intensity = step / n_steps  # Gradually increase intensity

            if drift_type == "seasonal":
                drifted = self.simulate_seasonal_drift(data, "winter")
            elif drift_type == "slang":
                drifted = self.simulate_slang_emergence(data, intensity)
            elif drift_type == "length":
                drifted = self.simulate_length_drift(data, 1 + intensity * 0.5)
            elif drift_type == "noise":
                drifted = self.simulate_noise_injection(data, intensity * 0.2)
            else:
                drifted = data.copy()

            drift_steps.append(drifted)

        return drift_steps


if __name__ == "__main__":
    # Quick test
    print("Testing Drift Simulator...\n")

    # Load validation data
    val_df = pd.read_csv("data/processed/val.csv")

    simulator = DriftSimulator(seed=42)

    # Test seasonal drift
    print("=" * 70)
    print("SIMULATING SEASONAL DRIFT (Winter)")
    print("=" * 70)
    winter_data = simulator.simulate_seasonal_drift(val_df.head(100), "winter")
    print(f"Original sample: {val_df.iloc[0]['text'][:100]}...")
    print(f"Modified sample: {winter_data.iloc[0]['text'][:100]}...")

    # Test slang emergence
    print("\n" + "=" * 70)
    print("SIMULATING SLANG EMERGENCE")
    print("=" * 70)
    slang_data = simulator.simulate_slang_emergence(val_df.head(100), intensity=0.3)
    print(f"Original sample: {val_df.iloc[5]['text'][:100]}...")
    print(f"Modified sample: {slang_data.iloc[5]['text'][:100]}...")

    # Test length drift
    print("\n" + "=" * 70)
    print("SIMULATING LENGTH DRIFT (1.5x longer)")
    print("=" * 70)
    longer_data = simulator.simulate_length_drift(val_df.head(10), change_factor=1.5)
    print(f"Original length: {len(val_df.iloc[0]['text'])} chars")
    print(f"Modified length: {len(longer_data.iloc[0]['text'])} chars")

    print("\n✅ Drift Simulator working!")
