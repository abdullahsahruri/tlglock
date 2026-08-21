import pytest

from tlglock.thfile import ThGate, ThNetwork, ThParseError, parse_th

from conftest import EQ5_TEXT, FIG3_TEXT, MULTI_TEXT


def test_parses_eq5(eq5):
    assert eq5.model == "eq5"
    assert eq5.inputs == ["X1", "X2", "X3"]
    assert eq5.outputs == ["Z"]
    assert len(eq5.gates) == 1
    g = eq5.gates[0]
    assert g.weights == [1, 1, 1]
    assert g.threshold == 3


def test_parses_fig3_seven_input_gate(fig3):
    """The .threshold line from Fig. 3 must split 2n+2 tokens correctly."""
    g = fig3.gates[0]
    assert g.inputs == ["K1", "X1", "X2", "X3", "Y2", "Y3", "K2"]
    assert g.output == "Z"
    assert g.weights == [3, 2, 1, 3, 2, 1, -2]
    assert g.threshold == 7
    assert g.fanin == 7


def test_negative_weights_survive_roundtrip(fig3):
    text = fig3.to_text()
    again = parse_th(text)
    assert again.gates[0].weights == fig3.gates[0].weights
    assert again.gates[0].threshold == fig3.gates[0].threshold
    assert again.to_text() == text


@pytest.mark.parametrize("src", [EQ5_TEXT, FIG3_TEXT, MULTI_TEXT])
def test_roundtrip_is_idempotent(src):
    once = parse_th(src).to_text()
    twice = parse_th(once).to_text()
    assert once == twice


def test_comments_and_blank_lines_ignored():
    net = parse_th(
        """
        # leading comment
        .model c   # trailing comment

        .input a b
        .output z
        .threshold a b z 1 1 2
        .end
        """
    )
    assert net.model == "c"
    assert len(net.gates) == 1


def test_line_continuation():
    net = parse_th(
        ".model c\n"
        ".input a b c\n"
        ".output z\n"
        ".threshold a b c \\\n"
        "  z 1 1 1 2\n"
    )
    assert net.gates[0].inputs == ["a", "b", "c"]
    assert net.gates[0].threshold == 2


def test_weight_of_and_fanin(fig3):
    g = fig3.gates[0]
    assert g.weight_of("K2") == -2
    assert g.weight_of("X3") == 3
    with pytest.raises(KeyError):
        g.weight_of("nope")


def test_fanout_count(multi):
    fo = multi.fanout_count()
    assert fo["n2"] == 2   # feeds both F and G
    assert fo["n1"] == 1
    assert fo["b"] == 1


def test_odd_token_count_rejected():
    with pytest.raises(ThParseError, match="2n\\+2"):
        parse_th(".model c\n.threshold a b z 1 1\n")


def test_non_integer_weight_rejected():
    with pytest.raises(ThParseError, match="non-integer"):
        parse_th(".model c\n.threshold a b z 1 x 2\n")


def test_mismatched_weights_rejected():
    with pytest.raises(ThParseError):
        ThGate(inputs=["a", "b"], output="z", weights=[1], threshold=1)


def test_repeated_input_on_gate_rejected():
    with pytest.raises(ThParseError, match="repeated"):
        ThGate(inputs=["a", "a"], output="z", weights=[1, 1], threshold=1)


def test_duplicate_primary_input_rejected():
    with pytest.raises(ThParseError, match="duplicate"):
        parse_th(".model c\n.input a a\n.output z\n.threshold a z 1 1\n")


def test_unknown_directive_rejected():
    with pytest.raises(ThParseError, match="unknown directive"):
        parse_th(".model c\n.latch a z\n")


def test_multiple_drivers_rejected():
    net = parse_th(
        ".model c\n.input a b\n.output z\n"
        ".threshold a z 1 1\n"
        ".threshold b z 1 1\n"
    )
    with pytest.raises(ThParseError, match="multiple drivers"):
        net.validate()


def test_undriven_signal_rejected():
    net = parse_th(".model c\n.input a\n.output z\n.threshold a q z 1 1 1\n")
    with pytest.raises(ThParseError, match="undriven"):
        net.validate()


def test_combinational_loop_rejected():
    net = ThNetwork(
        model="loop",
        inputs=["a"],
        outputs=["p"],
        gates=[
            ThGate(inputs=["a", "q"], output="p", weights=[1, 1], threshold=1),
            ThGate(inputs=["p"], output="q", weights=[1], threshold=1),
        ],
    )
    with pytest.raises(ThParseError, match="loop"):
        net.validate()


def test_gate_driving_primary_input_rejected():
    net = parse_th(".model c\n.input a\n.output a\n.threshold a a 1 1\n")
    with pytest.raises(ThParseError):
        net.validate()


def test_empty_input_rejected():
    with pytest.raises(ThParseError, match="empty"):
        parse_th("\n\n# nothing here\n")


def test_end_directive_stops_parsing():
    net = parse_th(
        ".model c\n.input a\n.output z\n.threshold a z 1 1\n.end\n"
        ".threshold a q 1 1\n"
    )
    assert len(net.gates) == 1


def test_copy_is_deep(multi):
    clone = multi.copy()
    clone.gates[0].weights[0] = 99
    clone.inputs.append("zzz")
    assert multi.gates[0].weights[0] == 1
    assert "zzz" not in multi.inputs


def test_topological_order_respects_dependencies(multi):
    order = [g.output for g in multi.topological_order()]
    assert order.index("n1") < order.index("F")
    assert order.index("n2") < order.index("F")
    assert order.index("n2") < order.index("G")
