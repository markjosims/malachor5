uv pip compile --universal --output-file requirements.txt requirements.in
uv pip compile --universal --output-file requirements-dev.txt requirements.in requirements-dev.in
uv pip compile --universal --output-file requirements-windows.txt requirements-windows.in requirements-dev.in