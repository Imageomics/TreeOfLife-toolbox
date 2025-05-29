#!/bin/bash

# Define variables
SOFTWARE_NAME="mongosh"
VERSION="1.10.5"  # Replace with the latest version available
INSTALL_DIR="$HOME/local/$SOFTWARE_NAME/$VERSION"
MODULEFILES_DIR="$HOME/local/share/lmodfiles/$SOFTWARE_NAME"
MODULE_FILE="$MODULEFILES_DIR/$VERSION.lua"

# Step 1: Download mongosh
echo "Downloading mongosh..."
wget -P $HOME/local/src https://downloads.mongodb.com/compass/mongosh-$VERSION-linux-x64.tgz

# Step 2: Extract mongosh
echo "Extracting mongosh..."
mkdir -p $INSTALL_DIR
tar -zxvf $HOME/local/src/mongosh-$VERSION-linux-x64.tgz -C $INSTALL_DIR --strip-components=1

# Step 3: Create a module file directory
echo "Setting up module file..."
mkdir -p $MODULEFILES_DIR

# Step 4: Create a module file using mkmod
echo "Loading mkmod module and creating mongosh module..."
module load mkmod
create_module.sh $SOFTWARE_NAME $VERSION $INSTALL_DIR

# Step 5: Add additional environment variables to the module (optional)
echo "Appending MONGOSH_HOME to module file..."
echo "setenv(\"MONGOSH_HOME\", \"$INSTALL_DIR\")" >> $MODULE_FILE

# Step 6: Add the modulefiles path to the environment and load the module
echo "Adding modulefiles to environment and loading mongosh module..."
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
echo "mongosh $VERSION installation is complete!"
echo "To use mongosh, ensure the module is loaded and run:"
echo "    mongosh"
