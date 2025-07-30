import torch


def torch_assert_close(actual, expected, **kwargs):
    """A wrapper around pytorch's assert_close, with a more helpful error message"""

    def make_msg(msg):
        return f"{msg}\n{actual=}\n{expected=}"

    torch.testing.assert_close(actual, expected, **kwargs, msg=make_msg)
