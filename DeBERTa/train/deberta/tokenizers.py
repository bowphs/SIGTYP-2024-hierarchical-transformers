#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#

""" tokenizers
"""

from .spm_tokenizer import *

__all__ = ['tokenizers']
tokenizers={
    'spm': SPMTokenizer,
    }
