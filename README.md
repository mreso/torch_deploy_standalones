Repro steps

    $ cd torch_deploy_standalones
    $ python3.8 -m venv transformers_venv
    $ source transformers_venv/bin/activate
    $ pip install -r requirements.txt

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_PREFIX_PATH="<pytorch_ROOT_DIR>" ..
    $ make -j

    $ ./already_in_use
    $ ./out_of_memory
