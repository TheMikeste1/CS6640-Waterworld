from unittest import TestCase

from agents.memory import Memory


class TestMemory(TestCase):
    def test_sample(self):
        mem = Memory(10)  # 10 should be more than enough for this test
        experiences = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

        mem.add(experiences[0])
        mem.add(experiences[1])
        mem.add(experiences[2])

        sample = mem.sample(3, zip_by_column=False)
        msg = (
            "When zip_by_column=False,"
            " experiences should be returned in their original format"
        )
        self.assertIn(experiences[0], sample, msg)
        self.assertIn(experiences[1], sample, msg)
        self.assertIn(experiences[2], sample, msg)

        sample = mem.sample(3, zip_by_column=True)
        msg = (
            "When zip_by_column=True,"
            " experiences should be returned by column. "
            "E.g. ((1, 4, 7), (2, 5, 8), (3, 6, 9))"
        )

        for row in range(len(sample[0])):
            t = sample[0][row], sample[1][row], sample[2][row]
            self.assertIn(t, experiences, msg)
