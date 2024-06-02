"""
Handsdown to generate documentation for the project.
"""
from handsdown.generators.base import BaseGenerator
from handsdown.generators.material import MaterialGenerator
from handsdown.utils.path_finder import PathFinder
from pathlib import Path

# this is our project root directory
repo_path = Path.cwd()

# this little tool works like `pathlib.Path.glob` with some extra magic
# but in this case `repo_path.glob("**/*.py")` would do as well
path_finder = PathFinder(repo_path)

# no docs for tests and build
path_finder.exclude("tests/*", "build/*")

# initialize generator
handsdown = MaterialGenerator(
    input_path=repo_path,
    output_path=repo_path / 'docs/source-code',
    source_paths=path_finder.glob("**/*.py"),
    source_code_url='https://github.com/Gorkowski/particula/blob/main/'
)

# generate all docs at once
handsdown.generate_docs()

# or generate just for one doc
# handsdown.generate_doc(repo_path / 'my_module' / 'source.py')

# generate index.md file
# handsdown.generate_index()

# and generate GitHub Pages and Read the Docs config files
# handsdown.generate_external_configs()

# navigate to `output` dir and check results
