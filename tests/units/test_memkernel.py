from memkernel.api import PostMemory
from memkernel.kernel import MemKernel


def test_memkernel():
    ker = MemKernel()
    to_ker = PostMemory("123", "I like Rust coding .")
    ker.remember(to_ker)
    temp = ker.recall("like")
    assert temp is not None
    assert temp.content.find("Rust") != -1
