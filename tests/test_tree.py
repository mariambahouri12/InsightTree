from domain.tree import Branch, EvidenceTree
from domain.enums import BranchStatus


def test_tree_adds_child():
    tree = EvidenceTree("Q0")
    child = Branch("Q1", tree.root.id, 1)
    tree.add_branch(child)
    assert child.id in tree.root.children_ids
    assert tree.get(child.id).question == "Q1"


def test_pruned_branch_is_not_explorable():
    branch = Branch("Q1", None, 1, status=BranchStatus.TERMINATED)
    assert not branch.is_explorable()
