"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

"""
Jet uses portions of Chromium V8.

Copyright 2008 the V8 project authors. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of Google Inc. nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import platform
import re
import os
import fnmatch

def guess_os():
    """
    Returns the name of the operating system.
    This will return 'linux' for Linux compatible system, 'macos' for Macs,
    'win32' for Windows, and 'freebsd' for FreeBSD.
    """
    id = platform.system()
    if id == 'Linux':
        return 'linux'
    elif id == 'Darwin':
        return 'macosx'
    elif id == 'Windows' or id == 'Microsoft':
        return 'win32'
    else:
        return None

def guess_word_size():
    """
    Returns the size of the pointer. For 64-bit systems, this will return '64',
    and '32' for 32-bit systems.
    """
    if '64' in platform.machine():
        return '64'
    else:
        archs = platform.architecture()
        for a in archs:
            if '64' in a:
                return '64'
        return '32'

def guess_arch():
    """
    Returns the architecture name of the system.
    """
    if is_windows():
        if guess_word_size() == '64':
            return 'x64'
        else:
            return 'win32'

    id = platform.machine()

    if is_mac():
        if guess_word_size() == '64' and id == 'i386':
            return 'x86_64'
        else:
            return id

    if id.startswith('arm'):
        return 'arm'
    elif (not id) or (not re.match('(x|i[3-6])86', id) is None):
        return id
    else:
        return None

def detect_num_cpus():
    """
    Detects the number of CPUs on a system. Cribbed from pp.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
            else: # OSX:
                return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
        if ncpus > 0:
            return ncpus
    return 1 # Default

def is_windows():
    """
    Returns True if you are using Windows.
    """
    return guess_os() == 'win32'

def is_windows64():
    """
    Returns True if you are using Visual Studio compiler in 64-bit mode.
    """
    if is_windows():
        return '64' in os.environ['LIB']
    else:
        return False

def is_unix():
    """
    Returns True if you are using Unix compatible system (Linux, Mac, and
    FreeBSD).
    """
    return not is_windows()

def is_mac():
    """
    Returns True if you are using Mac.
    """
    return guess_os() == 'macosx'

def is_linux():
    """
    Returns True if you are using Linux.
    """
    return guess_os() == 'linux'

def is64():
    """
    Returns True if running on 64-bit machine
    """
    return guess_word_size() == '64'

def navigate_all_files(root_path, patterns):
    """
    A generator function that iterates all files that matches the given patterns
    from the root_path.
    """
    for root, dirs, files in os.walk(root_path):
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                yield os.path.join(root, filename)

def get_all_files(root_path, patterns):
    """
    Returns a list of all files that matches the given patterns from the
    root_path.
    """
    ret = []
    for filepath in navigate_all_files(root_path, patterns):
        ret.append(filepath)
    return ret
