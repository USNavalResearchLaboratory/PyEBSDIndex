How to make a new release of ``PyEBSDIndex``
============================================

Create a PR to the ``main`` branch and go through the following steps.

Preparation
-----------
- Review the contributor list ``__credits__`` in ``pyebsdindex/__init__.py`` to ensure
  all contributors are included and sorted correctly.
- Bump ``__version__`` in ``pyebsdindex/__init__.py``, for example to "0.4.2".
- Update the changelog ``CHANGELOG.rst``.
- Let the PR collect comments for a day to ensure that other maintainers are
  comfortable with releasing. Merge.

Release (and tag)
-----------------
- Create a tagged, annotated (meaning with a release text) release draft with the name
  v0.4.2" and title "PyEBSDIndex 0.4.2". The tag target will be the ``main`` branch.
  Draw inspiration from previous release texts. Publish the release.
- Monitor the publish GitHub Action to ensure the release is successfully
  published to PyPI.

Post-release action
-------------------
- Monitor the `documentation build
  <https://readthedocs.org/projects/pyebsdindex/builds>`_ to ensure that the new stable
  documentation is successfully built from the release.
- Make a post-release PR to ``main`` with ``__version__`` updated (or reverted), e.g. to
  "0.4.dev0", and any updates to this guide if necessary.
- Tidy up GitHub issues.
