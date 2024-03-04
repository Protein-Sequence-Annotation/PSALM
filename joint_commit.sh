#!/bin/bash

# Check if a commit message is provided as an argument
if [ $# -eq 0 ]; then
    echo "Please provide a commit message."
    exit 1
fi

# Check if a username is provided as an argument
if [ $# -eq 1 ]; then
    if [ "$1" = "a" ]; then
        git config user.name "ArpanSarkar"
        git config user.email "arpan_sarkar@g.harvard.edu"
    else
        git config user.name "Kumaresh-Krishnan"
        git config user.email "kum2nan@gmail.com"
    fi
fi

# Commit the changes with the provided message
git commit -m "$1


Co-authored-by: ArpanSarkar <arpan_sarkar@g.harvard.edu>
Co-authored-by: Kumaresh-Krishnan <kum2nan@gmail.com>"

# Push the changes to the remote repository
git push

echo "Joint commit successful."
