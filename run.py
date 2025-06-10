#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

from tritonparse.common import is_fbcode
from tritonparse.utils import init_parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    if is_fbcode():
        from tritonparse.fb.utils import fb_parse as parse
    else:
        from tritonparse.utils import oss_parse as parse
    parse(args)


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
