"""Tests for ai.npu.latency.conv_split on synthetic conv tasks."""

from test_npu_trace import _pad, _word, _write_log

from ai.npu.latency import conv_split


def _conv(kernel, width, in_h, out_h, pad_top, feature, weight=0x1000):
    """Return the command words of a conv task as _write_log expects."""
    words = [
        _word(0x201, conv_split.CNA_DATA_SIZE0, (width << 16) | in_h),
        _word(
            0x201, conv_split.CNA_WEIGHT_SIZE2, (kernel << 24) | (kernel << 16)
        ),
        _word(0x201, conv_split.CNA_PAD_CON0, 0x10 | pad_top),
        _word(0x201, conv_split.CNA_FEATURE_ADDR, feature),
        _word(0x201, conv_split.CNA_DCOMP_ADDR0, weight),
        _word(
            0x801, conv_split.CORE_DATAOUT_SIZE0, ((out_h - 1) << 16) | width
        ),
    ]
    return ("0x1d", _pad(words))


def _log(tmp_path, tasks):
    return _write_log(
        tmp_path, [("0x7", {2: [tasks[0]], 3: [tasks[1]], 4: [tasks[2]]})]
    )


def test_3x3_split_reads_halo_rows(tmp_path):
    row = 20 * 16
    log = _log(
        tmp_path,
        [
            _conv(3, 20, 8, 7, 1, 0),
            _conv(3, 20, 9, 7, 0, 6 * row),
            _conv(3, 20, 7, 6, 0, 13 * row),
        ],
    )
    line = conv_split.split_lines(log)[1]
    assert "kernel=3x3" in line
    assert "in_h=(8, 9, 7) out_h=(7, 7, 6)" in line
    assert "pad_top=(1, 0, 0) first_row=(0, 6, 13) weights=1" in line


def test_1x1_split_has_no_halo(tmp_path):
    row = 20 * 16
    log = _log(
        tmp_path,
        [
            _conv(1, 20, 7, 7, 0, 0),
            _conv(1, 20, 7, 7, 0, 7 * row),
            _conv(1, 20, 6, 6, 0, 14 * row),
        ],
    )
    line = conv_split.split_lines(log)[1]
    assert "in_h=(7, 7, 6) out_h=(7, 7, 6)" in line


def test_channel_split_has_one_input_and_three_weights(tmp_path):
    log = _log(
        tmp_path,
        [_conv(3, 40, 40, 20, 1, 0, w) for w in (0x1000, 0x2000, 0x3000)],
    )
    line = conv_split.split_lines(log)[1]
    assert "first_row=(0, 0, 0) weights=3" in line
