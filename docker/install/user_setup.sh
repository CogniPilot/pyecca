#!/bin/bash

# setup .profile, note bashrc doesn't get sourced by docker by defualt, .profile does
echo "source ~/.venv/bin/activate" >> ~/.profile
