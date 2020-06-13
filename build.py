#!/usr/bin/env python

import argparse
import shutil
import subprocess
import os
from pathlib import Path


def system_call(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    if subprocess.call(cmd) != 0:
        raise RuntimeError(f'Failed to execute {cmd}')


def build_pdoc(cwd: Path, jet_path: Path) -> None:
    print('Building pyjet and pdoc...')
    system_call(f'pip install -U {jet_path}')
    system_call('pdoc pyjet --html --force')
    shutil.move('html/pyjet.html', str(cwd / 'pdoc' / 'index.html'))


def build_doxygen(cwd: Path, jet_path: Path) -> None:
    print('Building doxygen...')
    shutil.rmtree(str(cwd / 'doxygen'), ignore_errors=True)
    doxygen_path = jet_path / 'doc' / 'doxygen'
    system_call(f'./build-doxygen.sh {doxygen_path}')
    shutil.move(str(doxygen_path / 'html'), str(cwd / 'doxygen'))


def build_website(gh_pages_path: Path) -> None:
    print('Building website...')
    system_call('bundle exec jekyll build')
    system_call(f'rsync -ar _site/ {gh_pages_path}/')


def copy_website(gh_pages_path: Path) -> None:
    print('Copying website...')
    system_call(f'rsync -ar _site/ {gh_pages_path}/')


def main(enable_pdoc: bool, enable_doxygen: bool, enable_website: bool, jet_path: Path, gh_pages_path: Path) -> None:
    cwd = Path(os.getcwd())

    if enable_pdoc:
        build_pdoc(cwd, jet_path)
    if enable_doxygen:
        build_doxygen(cwd, jet_path)
    if enable_website:
        build_website(gh_pages_path)


if __name__ == '__main__':
    home = Path(os.environ['HOME'])
    codes = home / 'Codes'
    default_jet_path = codes / 'fluid-engine-dev-main'
    default_gh_pages_path = codes / 'fluid-engine-dev-gh-pages'

    parser = argparse.ArgumentParser(description='Build website.')
    parser.add_argument('--build-pdoc', action='store_true')
    parser.add_argument('--build-doxygen', action='store_true')
    parser.add_argument('--build-website', action='store_true')
    parser.add_argument('--build-all', action='store_true')
    parser.add_argument('--jet-path', type=str, default=str(default_jet_path))
    parser.add_argument('--gh-pages-path', type=str, default=str(default_gh_pages_path))

    args = parser.parse_args()

    enable_pdoc = args.build_pdoc or args.build_all
    enable_doxygen = args.build_doxygen or args.build_all
    enable_website = args.build_website or args.build_all

    main(enable_pdoc, enable_doxygen, enable_website, Path(args.jet_path), Path(args.gh_pages_path))
