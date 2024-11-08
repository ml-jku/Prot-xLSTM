
# Original code from ProtMamba under Apache License 2.0.

from protxlstm.utils import MASK_TO_ID, AA_TO_ID
import numpy as np

class AbstractFIM(object):
    def __init__(self,
                 max_patches=5,
                 mask_fraction=0.2,
                 always_mask=False,
                 mask_tokens=MASK_TO_ID,
                 eos_token=AA_TO_ID["<eos>"],
                 add_position_ids=False,
                 troubleshoot=False):
        """
        This class is designed to concatenate sequences based on different scrambling strategies.
        It takes a list of sequences, tuples indicating the start and end indices of each sequence,
        an optional number of patches to sample, and a scrambling strategy as inputs.
        """
        self.troubleshoot = troubleshoot
        self.max_patches = max_patches
        self.mask_fraction = mask_fraction
        self.mask_tokens = mask_tokens
        assert len(
            self.mask_tokens) >= self.max_patches, "Number of mask tokens must be bigger than max number of patches."
        self.eos_token = eos_token
        self.add_position_ids = add_position_ids
        self.always_mask = always_mask

    def apply(self, sequences, tuples):
        """
        This function concatenates the sequences scrambling each one according to the scrambling strategy.
        """
        input_ids, position_ids = [], []
        for t in tuples:
            seq, pos = self.fim(sequences, t)
            input_ids.extend(seq)
            if self.add_position_ids:
                position_ids.extend(pos)
        if self.add_position_ids:
            return input_ids, position_ids
        return input_ids, None

    def fim(self, sequences, t):
        """
        This function concatenates the sequence's parts based on the scrambling strategy.
        """
        raise NotImplementedError


class NoFIM(AbstractFIM):
    def __init__(self,
                 max_patches=5,
                 mask_fraction=0.2,
                 always_mask=False,
                 mask_tokens=MASK_TO_ID,
                 eos_token=AA_TO_ID["<eos>"],
                 add_position_ids=False,
                 troubleshoot=False):
        super().__init__(max_patches, mask_fraction, always_mask, mask_tokens, eos_token, add_position_ids, troubleshoot)

    def fim(self, sequences, t):
        """
        This function keeps the sequence identical without any scrambling.
        """
        if self.add_position_ids:
            position_ids = np.arange(t[0], t[1]) - t[0]
            return sequences[t[0]:t[1]], position_ids
        return sequences[t[0]:t[1]], None


class SingleSpanFIM(AbstractFIM):

    def __init__(self,
                 max_patches=5,
                 mask_fraction=0.2,
                 always_mask=False,
                 mask_tokens=MASK_TO_ID,
                 eos_token=AA_TO_ID["<eos>"],
                 add_position_ids=False,
                 troubleshoot=False):
        super().__init__(max_patches, mask_fraction, always_mask, mask_tokens, eos_token, add_position_ids, troubleshoot)

    def fim(self, sequences, t):
        """
        This function creates and concatenates parts of the sequences based on the OpenAI scrambling strategy.
        It randomly selects two indices within the range of the given tuple,
        splits the sequence into three parts based on these indices, and then concatenates them with the
        masked patch at the end
        """
        new_tuple = tuple(np.sort(np.random.choice(np.arange(t[0] + 1, t[1]), 2, replace=False)))
        part1 = sequences[t[0]:new_tuple[0]]
        part2 = sequences[new_tuple[0]:new_tuple[1]]
        part3 = sequences[new_tuple[1]:t[1]]
        sequence = np.concatenate([part1, [self.mask_tokens["<mask-1>"]], part3, [self.mask_tokens["<mask-1>"]], part2])
        position_ids_sequence = None
        if self.add_position_ids:
            position_ids = np.arange(t[0], t[1]) - t[0]
            position_ids_part1 = position_ids[t[0]:new_tuple[0]]
            position_ids_part2 = position_ids[new_tuple[0]:new_tuple[1]]
            position_ids_part3 = position_ids[new_tuple[1]:t[1]]
            position_ids_sequence = np.concatenate(
                [position_ids_part1, [position_ids_part2[0]], position_ids_part3, [position_ids_part2[0]],
                 position_ids_part2])

        return sequence, position_ids_sequence


class MultipleSpanFIM(AbstractFIM):
    def __init__(self,
                 max_patches=5,
                 mask_fraction=0.2,
                 always_mask=False,
                 mask_tokens=MASK_TO_ID,
                 eos_token=AA_TO_ID["<eos>"],
                 add_position_ids=False,
                 troubleshoot=False):
        super().__init__(max_patches, mask_fraction, always_mask, mask_tokens, eos_token, add_position_ids, troubleshoot)

    def fim(self, sequences, t):
        """
        This function creates and concatenates parts of the sequences based on the inpaint scrambling strategy.
        It randomly selects `2*num_patches` indices within the range of the given tuple,
        splits the sequence into unmasked and masked parts based on these indices, and then concatenates them.
        The number of patches is sampled from a poisson distribution with upper limit `self.max_patches` and average 1.
        The concatenation is done by joining all unmaksed parts (interleaved with mask tokens) and afterwards
        all masked parts (interleaved with mask tokens). At the end of the unmasked parts, a special token is added
        to indicate the end of the unmasked parts, and at the end of the masked parts, a special token is added
        to indicate the end of the masked parts.
        """
        # sample num_patches from a discrete poisson distribution with upper limit L
        def sample_lengths(start, end):
            """
            Sample a length uniformly from 1 to max_L*self.mask_fraction (must be bigger than 1).
            If the length is larger than max_L, return max_L.
            """
            max_L = end - start
            length = np.random.randint(1, max(int(max_L * self.mask_fraction), 2))
            return min(length, max_L)

        # sample num_patches from a discrete poisson distribution with upper limit max_patches
        num_patches = 1000
        while num_patches > self.max_patches:
            num_patches = np.random.poisson(1)
        if self.always_mask:
            num_patches = max(num_patches, 1)
        # sample num_patches starting points for the masked positions (+ final position)
        start_patches = list(np.sort(np.random.choice(np.arange(t[0] + 1, t[1]),
                                                      num_patches,
                                                      replace=False))) + [t[1]]
        # sample num_patches lengths of the patches
        len_patches = [sample_lengths(start_patches[i], start_patches[i + 1])
                       for i in range(len(start_patches) - 1)]
        # create masked tuples with start and end indices of the patches
        masked_tuples = [(start_patches[i], start_patches[i] + len_patches[i]) for i in range(len(start_patches) - 1)]
        # split the sequences into unmasked and masked parts
        unmasked_sequence, masked_sequence, unmasked_position_ids, masked_position_ids = self.split_sequences(sequences,
                                                                                                              t,
                                                                                                              masked_tuples)

        if self.troubleshoot:
            print(f"For sequence in {t}: sampled {num_patches=}, {start_patches=}, {len_patches=}, {masked_tuples=}")
        # concatenate the unmasked and masked parts
        return unmasked_sequence + masked_sequence, unmasked_position_ids + masked_position_ids if self.add_position_ids else None

    def split_sequences(self, sequences, t, masked_tuples):
        """
        This function splits the sequences into unmasked and masked parts based on the given tuples.
        Args:
            t (tuple): The start and end index of each sequence.
            masked_tuples (list): A list of tuples specifying the indices for masked regions.
        Returns:
            unmasked_parts (list): The unmasked parts of the sequences interleaved with mask_tokens.
            masked_parts (list): The masked parts of the sequences interleaved with mask_tokens.
        """
        unmasked_parts, masked_parts = [], []
        unmasked_positions, masked_positions = [], []
        position_ids = None
        start, end = t
        if self.add_position_ids:
            position_ids = np.arange(start, end) - start
        for i, region in enumerate(masked_tuples):
            mask_token = self.mask_tokens[f"<mask-{i + 1}>"]
            unmasked_parts.extend(sequences[start:region[0]])
            unmasked_parts.append(mask_token)
            masked_parts.append(mask_token)
            masked_parts.extend(sequences[region[0]:region[1]])
            if self.add_position_ids:
                unmasked_positions.extend(position_ids[start-t[0]:region[0]-t[0]])
                unmasked_positions.append(position_ids[region[0]-t[0]])
                masked_positions.append(position_ids[region[0]-t[0]])
                masked_positions.extend(position_ids[region[0]-t[0]:region[1]-t[0]])

            start = region[1]
        unmasked_parts.extend(sequences[start:end])
        if self.add_position_ids:
            unmasked_positions.extend(position_ids[start-t[0]:end-t[0]])
        if len(masked_tuples) > 0:
            unmasked_parts.append(self.eos_token)
            if self.add_position_ids:
                unmasked_positions.append(0)
        return unmasked_parts, masked_parts, unmasked_positions, masked_positions
