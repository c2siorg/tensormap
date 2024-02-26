# Check if the "data" folder exists
if [ ! -d "data" ]; then
    # If not, create the "data" folder
    mkdir data
fi

# Update Conda environment
conda env update --file environment.yml --prune

