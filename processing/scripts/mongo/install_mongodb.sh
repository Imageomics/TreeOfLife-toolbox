#!/bin/bash

# Define variables
SOFTWARE_NAME="mongodb"
VERSION="8.0.4"
INSTALL_DIR="$HOME/local/$SOFTWARE_NAME/$VERSION"
MODULEFILES_DIR="$HOME/local/share/lmodfiles/$SOFTWARE_NAME"
MODULE_FILE="$MODULEFILES_DIR/$VERSION.lua"
BASE_DIR=""  # Set your base path for data
DATA_DIR="$BASE_DIR/mongodb_data"  # Updated data directory

# Step 1: Download MongoDB (optional if already downloaded)
# Uncomment the wget line if the tarball is not already downloaded.
echo "Downloading MongoDB..."
wget -P $HOME/local/src https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-rhel93-8.0.4.tgz

# Step 2: Extract the tarball
echo "Extracting MongoDB..."
mkdir -p $INSTALL_DIR
tar -zxvf $HOME/local/src/mongodb-linux-x86_64-rhel93-8.0.4.tgz -C $INSTALL_DIR --strip-components=1

# Step 3: Create the data directory for MongoDB
echo "Creating data directory for MongoDB at $DATA_DIR..."
mkdir -p $DATA_DIR
chmod -R 770 $DATA_DIR  # Allow group access
chgrp -R PAS2136 $DATA_DIR  # Set group ownership (replace 'PAS2136' with your actual group name)
chmod g+s $DATA_DIR  # Ensure new files inherit the group

# Step 4: Create a module file directory
echo "Setting up module file..."
mkdir -p $MODULEFILES_DIR

# Step 5: Create a module file manually
echo "Creating MongoDB module file..."
cat > $MODULE_FILE <<EOL
-- MongoDB Module
help([[
MongoDB $VERSION
]])

whatis("Version: $VERSION")
whatis("Description: MongoDB $VERSION")

prepend_path("PATH", "$INSTALL_DIR/bin")
setenv("MONGODB_DATA", "$DATA_DIR")
EOL

# Step 6: Add the modulefiles path to the environment
echo "Adding modulefiles to environment..."
if ! grep -q "module use $HOME/local/share/lmodfiles" $HOME/.bashrc; then
  echo "module use $HOME/local/share/lmodfiles" >> $HOME/.bashrc
fi
if ! grep -q "module load $SOFTWARE_NAME/$VERSION" $HOME/.bashrc; then
  echo "module load $SOFTWARE_NAME/$VERSION" >> $HOME/.bashrc
fi

# Step 7: Reload shell and print instructions
echo "Reloading shell configuration..."
source $HOME/.bashrc

# Step 8: Display success message
echo "MongoDB $VERSION installation is complete!"
echo "Data directory: $DATA_DIR"
echo "To start MongoDB, use the following command:"
echo "    module load $SOFTWARE_NAME/$VERSION"
echo "    mongod --dbpath=$DATA_DIR"
