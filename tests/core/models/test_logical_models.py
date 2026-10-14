"""Tests for the logical model: BlockSpan, LogicalNode, LogicalDocument."""

import dataclasses

import pytest

from document_processor.core.models.logical import BlockSpan, ChildRef, LogicalNode
from document_processor.core.models.ranges import TextRange
from tests.factories import (
    make_block_span,
    make_child_ref,
    make_logical_document,
    make_logical_node,
)


def node(node_id: str, *children: str) -> LogicalNode:
    return make_logical_node(id=node_id, content=tuple(make_child_ref(c) for c in children))


def tree_nodes() -> tuple[LogicalNode, ...]:
    """Two roots, in pre-order:

        chuong-1
          dieu-1
            khoan-1
            khoan-2
          dieu-2
        chuong-2
    """
    return (
        node("chuong-1", "dieu-1", "dieu-2"),
        node("dieu-1", "khoan-1", "khoan-2"),
        node("khoan-1"),
        node("khoan-2"),
        node("dieu-2"),
        node("chuong-2"),
    )


ROOTS = ("chuong-1", "chuong-2")


# --------------------------------------------------------------------------- #
# BlockSpan
# --------------------------------------------------------------------------- #


class TestBlockSpan:
    def test_valid(self) -> None:
        span = make_block_span(block_id="pb3")
        assert span.block_id == "pb3"

    @pytest.mark.parametrize("block_id", ["", " ", "\t"])
    def test_empty_block_id_rejected(self, block_id: str) -> None:
        with pytest.raises(ValueError, match="block_id"):
            make_block_span(block_id=block_id)

    def test_frozen(self) -> None:
        span = make_block_span()
        with pytest.raises(dataclasses.FrozenInstanceError):
            span.block_id = "pb2"  # type: ignore[misc]

    def test_keyword_only(self) -> None:
        from document_processor.core.models.logical import BlockSpan
        from document_processor.core.models.ranges import TextRange

        with pytest.raises(TypeError):
            BlockSpan("pb1", TextRange(0, 1))  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# LogicalNode
# --------------------------------------------------------------------------- #


class TestLogicalNode:
    def test_valid(self) -> None:
        n = make_logical_node()
        assert n.id == "dieu-1"

    @pytest.mark.parametrize("node_id", ["", "  "])
    def test_empty_id_rejected(self, node_id: str) -> None:
        with pytest.raises(ValueError, match="Empty LogicalNode id"):
            make_logical_node(id=node_id)

    @pytest.mark.parametrize("level", ["", "  "])
    def test_empty_level_rejected(self, level: str) -> None:
        with pytest.raises(ValueError, match="'dieu-1'.*'level'"):
            make_logical_node(level=level)

    def test_number_none_allowed(self) -> None:
        assert make_logical_node(number=None).number is None

    @pytest.mark.parametrize("number", ["1", "a", "đ", "IV", "1a"])
    def test_number_kept_as_written(self, number: str) -> None:
        assert make_logical_node(number=number).number == number

    @pytest.mark.parametrize("number", ["", " "])
    def test_empty_number_rejected(self, number: str) -> None:
        with pytest.raises(ValueError, match="'dieu-1'.*'number'"):
            make_logical_node(number=number)

    def test_marker_none_allowed(self) -> None:
        # An auto-numbered khoản has no marker text in the block.
        assert make_logical_node(marker=None).marker is None

    def test_node_without_marker_title_or_content_allowed(self) -> None:
        n = make_logical_node(marker=None, title=(), content=())
        assert n.content == ()
        assert n.children_ids == ()

    def test_duplicate_children_rejected(self) -> None:
        with pytest.raises(ValueError, match="'dieu-1'.*duplicate children ids: khoan-1"):
            make_logical_node(
                content=(
                    make_child_ref("khoan-1"),
                    make_block_span(),
                    make_child_ref("khoan-2"),
                    make_child_ref("khoan-1"),
                )
            )

    def test_frozen(self) -> None:
        n = make_logical_node()
        with pytest.raises(dataclasses.FrozenInstanceError):
            n.id = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ChildRef
# --------------------------------------------------------------------------- #


class TestChildRef:
    def test_valid(self) -> None:
        assert ChildRef(node_id="dieu-1").node_id == "dieu-1"

    @pytest.mark.parametrize("node_id", ["", " ", "\t"])
    def test_blank_node_id_rejected(self, node_id: str) -> None:
        with pytest.raises(ValueError, match="node_id"):
            ChildRef(node_id=node_id)

    def test_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            ChildRef("dieu-1")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        ref = ChildRef(node_id="dieu-1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.node_id = "x"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        assert ChildRef(node_id="a") == ChildRef(node_id="a")
        assert ChildRef(node_id="a") != ChildRef(node_id="b")


# --------------------------------------------------------------------------- #
# LogicalNode content
# --------------------------------------------------------------------------- #


def text(block_id: str) -> BlockSpan:
    return make_block_span(block_id=block_id, text_range=TextRange(0, 5))


class TestLogicalNodeContent:
    def test_only_text(self) -> None:
        n = make_logical_node(content=(text("pb1"), text("pb2")))
        assert n.children_ids == ()

    def test_only_children(self) -> None:
        n = make_logical_node(content=(make_child_ref("a"), make_child_ref("b")))
        assert n.children_ids == ("a", "b")

    def test_interleaved_content_kept_in_order(self) -> None:
        # lead-in, list, text between, second list, closing remark
        content = (
            text("pb1"),
            make_child_ref("item-1"),
            make_child_ref("item-2"),
            text("pb4"),
            text("pb5"),
            make_child_ref("item-3"),
            text("pb7"),
        )
        n = make_logical_node(content=content)
        assert n.content == content
        assert n.children_ids == ("item-1", "item-2", "item-3")

    def test_children_ids_is_a_tuple(self) -> None:
        n = make_logical_node(content=(make_child_ref("a"),))
        assert isinstance(n.children_ids, tuple)

    def test_children_ids_not_a_constructor_argument(self) -> None:
        with pytest.raises(TypeError):
            make_logical_node(_children_ids=("a",))

    def test_children_ids_not_in_repr(self) -> None:
        assert "_children_ids" not in repr(make_logical_node(content=(make_child_ref("a"),)))

    def test_same_block_may_appear_twice_in_content(self) -> None:
        # Two spans of one block (split by the builder) are allowed; only children must be unique.
        a = make_block_span(block_id="pb1", text_range=TextRange(0, 3))
        b = make_block_span(block_id="pb1", text_range=TextRange(4, 9))
        n = make_logical_node(content=(a, b))
        assert n.content == (a, b)

    def test_equal_nodes_compare_equal(self) -> None:
        content = (text("pb1"), make_child_ref("a"))
        assert make_logical_node(content=content) == make_logical_node(content=content)

    def test_content_order_matters_for_equality(self) -> None:
        a = make_logical_node(content=(text("pb1"), make_child_ref("a")))
        b = make_logical_node(content=(make_child_ref("a"), text("pb1")))
        assert a != b


# --------------------------------------------------------------------------- #
# LogicalDocument: valid documents
# --------------------------------------------------------------------------- #


class TestLogicalDocumentValid:
    def test_no_nodes(self) -> None:
        doc = make_logical_document()
        assert doc.nodes == ()
        assert doc.root_ids == ()

    def test_no_nodes_with_front_matter(self) -> None:
        # A document with no recognized headings keeps everything in front matter.
        doc = make_logical_document(front_matter=(make_block_span(),))
        assert len(doc.front_matter) == 1

    def test_single_root(self) -> None:
        doc = make_logical_document(root_ids=("dieu-1",), nodes=(node("dieu-1"),))
        assert doc.get_node("dieu-1").id == "dieu-1"

    def test_tree_with_several_roots(self) -> None:
        doc = make_logical_document(root_ids=ROOTS, nodes=tree_nodes())
        assert [n.id for n in doc.nodes] == [
            "chuong-1",
            "dieu-1",
            "khoan-1",
            "khoan-2",
            "dieu-2",
            "chuong-2",
        ]

    def test_flat_roots(self) -> None:
        nodes = (node("dieu-1"), node("dieu-2"), node("dieu-3"))
        doc = make_logical_document(root_ids=("dieu-1", "dieu-2", "dieu-3"), nodes=nodes)
        assert len(doc.nodes) == 3


# --------------------------------------------------------------------------- #
# LogicalDocument: invalid documents
# --------------------------------------------------------------------------- #


class TestLogicalDocumentInvalid:
    def test_duplicate_node_id(self) -> None:
        nodes = (node("dieu-1"), node("dieu-1"))
        with pytest.raises(ValueError, match="duplicate node id 'dieu-1' at position 1"):
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)

    def test_dangling_root_id(self) -> None:
        with pytest.raises(ValueError, match="no node with id dieu-9"):
            make_logical_document(root_ids=("dieu-1", "dieu-9"), nodes=(node("dieu-1"),))

    def test_dangling_child_id(self) -> None:
        nodes = (node("dieu-1", "khoan-1", "khoan-9"), node("khoan-1"))
        with pytest.raises(ValueError, match="no node with id khoan-9"):
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)

    def test_root_ids_given_but_no_nodes(self) -> None:
        with pytest.raises(ValueError, match="no node with id dieu-1"):
            make_logical_document(root_ids=("dieu-1",))

    def test_duplicate_root_id(self) -> None:
        with pytest.raises(ValueError, match="'dieu-1' already visited"):
            make_logical_document(root_ids=("dieu-1", "dieu-1"), nodes=(node("dieu-1"),))

    def test_node_with_two_parents(self) -> None:
        nodes = (
            node("dieu-1", "khoan-1"),
            node("khoan-1"),
            node("dieu-2", "khoan-1"),
        )
        with pytest.raises(ValueError, match="'khoan-1' already visited"):
            make_logical_document(root_ids=("dieu-1", "dieu-2"), nodes=nodes)

    def test_root_that_is_also_a_child(self) -> None:
        nodes = (node("dieu-1", "khoan-1"), node("khoan-1"))
        with pytest.raises(ValueError, match="'khoan-1' already visited"):
            make_logical_document(root_ids=("dieu-1", "khoan-1"), nodes=nodes)

    def test_node_is_its_own_child(self) -> None:
        with pytest.raises(ValueError, match="'dieu-1' already visited"):
            make_logical_document(root_ids=("dieu-1",), nodes=(node("dieu-1", "dieu-1"),))

    def test_cycle_reachable_from_root(self) -> None:
        nodes = (node("dieu-1", "khoan-1"), node("khoan-1", "dieu-1"))
        with pytest.raises(ValueError, match="'dieu-1' already visited"):
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)

    def test_cycle_not_reachable_from_root(self) -> None:
        nodes = (node("dieu-1"), node("a", "b"), node("b", "a"))
        with pytest.raises(ValueError, match="unreachable") as exc:
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)
        assert "'a'" in str(exc.value)
        assert "'b'" in str(exc.value)

    def test_orphan_node(self) -> None:
        nodes = (node("dieu-1"), node("khoan-7"))
        with pytest.raises(ValueError, match="unreachable.*'khoan-7'"):
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)

    def test_nodes_without_any_root(self) -> None:
        with pytest.raises(ValueError, match="unreachable"):
            make_logical_document(nodes=(node("dieu-1"),))

    def test_siblings_out_of_order(self) -> None:
        nodes = (node("dieu-1", "khoan-1", "khoan-2"), node("khoan-2"), node("khoan-1"))
        with pytest.raises(ValueError, match="not in pre-order"):
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)

    def test_roots_out_of_order(self) -> None:
        nodes = (node("dieu-2"), node("dieu-1"))
        with pytest.raises(ValueError, match="not in pre-order"):
            make_logical_document(root_ids=("dieu-1", "dieu-2"), nodes=nodes)

    def test_post_order_rejected(self) -> None:
        nodes = (node("khoan-1"), node("khoan-2"), node("dieu-1", "khoan-1", "khoan-2"))
        with pytest.raises(ValueError, match="not in pre-order"):
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)

    def test_breadth_first_order_rejected(self) -> None:
        # Level by level: both articles before any clause.
        nodes = (
            node("chuong-1", "dieu-1", "dieu-2"),
            node("dieu-1", "khoan-1"),
            node("dieu-2"),
            node("khoan-1"),
        )
        with pytest.raises(ValueError, match="not in pre-order"):
            make_logical_document(root_ids=("chuong-1",), nodes=nodes)


# --------------------------------------------------------------------------- #
# LogicalDocument: lookup
# --------------------------------------------------------------------------- #


class TestLogicalDocumentLookup:
    def test_get_node_returns_the_same_object(self) -> None:
        nodes = tree_nodes()
        doc = make_logical_document(root_ids=ROOTS, nodes=nodes)
        for n in nodes:
            assert doc.get_node(n.id) is n

    def test_get_node_missing_raises_key_error(self) -> None:
        doc = make_logical_document(root_ids=ROOTS, nodes=tree_nodes())
        with pytest.raises(KeyError, match="no node with id dieu-9"):
            doc.get_node("dieu-9")

    def test_get_node_on_empty_document(self) -> None:
        with pytest.raises(KeyError):
            make_logical_document().get_node("dieu-1")

    def test_mapping_is_read_only(self) -> None:
        doc = make_logical_document(root_ids=("dieu-1",), nodes=(node("dieu-1"),))
        with pytest.raises(TypeError):
            doc._node_by_id_mapping["x"] = node("x")  # type: ignore[index]

    def test_mapping_not_a_constructor_argument(self) -> None:
        with pytest.raises(TypeError):
            make_logical_document(_node_by_id_mapping={})

    def test_mapping_not_in_repr(self) -> None:
        doc = make_logical_document(root_ids=("dieu-1",), nodes=(node("dieu-1"),))
        assert "_node_by_id_mapping" not in repr(doc)

    def test_equal_documents_compare_equal(self) -> None:
        a = make_logical_document(root_ids=ROOTS, nodes=tree_nodes())
        b = make_logical_document(root_ids=ROOTS, nodes=tree_nodes())
        assert a == b

    def test_frozen(self) -> None:
        doc = make_logical_document()
        with pytest.raises(dataclasses.FrozenInstanceError):
            doc.id = "x"  # type: ignore[misc]


class TestLogicalDocumentWithInterleavedContent:
    def test_pre_order_follows_content_order(self) -> None:
        nodes = (
            make_logical_node(
                id="dieu-1",
                content=(
                    text("pb1"),
                    make_child_ref("khoan-1"),
                    text("pb3"),
                    make_child_ref("khoan-2"),
                ),
            ),
            node("khoan-1"),
            node("khoan-2"),
        )
        doc = make_logical_document(root_ids=("dieu-1",), nodes=nodes)
        assert [n.id for n in doc.nodes] == ["dieu-1", "khoan-1", "khoan-2"]

    def test_wrong_order_against_content_rejected(self) -> None:
        nodes = (
            make_logical_node(
                id="dieu-1",
                content=(make_child_ref("khoan-1"), text("pb2"), make_child_ref("khoan-2")),
            ),
            node("khoan-2"),
            node("khoan-1"),
        )
        with pytest.raises(ValueError, match="not in pre-order"):
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)

    def test_dangling_child_ref_rejected(self) -> None:
        nodes = (make_logical_node(id="dieu-1", content=(text("pb1"), make_child_ref("khoan-9"))),)
        with pytest.raises(ValueError, match="no node with id khoan-9"):
            make_logical_document(root_ids=("dieu-1",), nodes=nodes)