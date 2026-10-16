# Installation


<span style="font-size:1.3em; font-weight:bold">Recommended: Create a Virtual Environment</span>

Using Conda:

    conda create -n hermes python=3.12
    conda activate hermes


<span style="font-size:1.3em; font-weight:bold">Install hermes</span>

Hermes is not on PyPI; the `hermes` name there belongs to an unrelated project. Install from source:

    git clone https://github.com/odinzen/hermes.git
    cd hermes
    python -m pip install -e .
