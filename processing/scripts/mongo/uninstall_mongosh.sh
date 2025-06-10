#!/bin/bash

# Define variables
SOFTWARE_NAME="mongosh"
VERSION="1.10.5"  # Ensure this matches the version installed
INSTALL_DIR="$HOME/local/$SOFTWARE_NAME/$VERSION"
MODULEFILES_DIR="$HOME/local/share/lmodfiles/$SOFTWARE_NAME"
MODULE_FILE="$MODULEFILES_DIR/$VERSION.lua"

# Step 1: Stop using the module
echo "Unloading the mongosh module (if loaded)..."
module unload $SOFTWARE_NAME/$VERSION

# Step 2: Remove the mongosh installation directory
echo "Removing mongosh installation directory..."
if [ -d "$INSTALL_DIR" ]; then
  rm -rf "$INSTALL_DIR"
  echo "Installation directory removed: $INSTALL_DIR"
else
  echo "Installation directory not found: $INSTALL_DIR"
fi

# Step 3: Remove the module file
echo "Removing mongosh module file..."
if [ -f "$MODULE_FILE" ]; then
  rm -f "$MODULE_FILE"
  echo "Module file removed: $MODULE_FILE"
else
  echo "Module file not found: $MODULE_FILE"
fi

# Step 4: Remove the module directory if empty
if [ -d "$MODULEFILES_DIR" ] && [ -z "$(ls -A $MODULEFILES_DIR)" ]; then
  echo "Module directory is empty. Removing..."
  rmdir "$MODULEFILES_DIR"
  echo "Module directory removed: $MODULEFILES_DIR"
else
  echo "Module directory is not empty or does not exist: $MODULEFILES_DIR"
fi

# Step 5: Clean up environment variables in .bashrc
echo "Cleaning up environment variables in .bashrc..."
sed -i '/module use $HOME\/local\/share\/lmodfiles/d' $HOME/.bashrc
sed -i "/module load $SOFTWARE_NAME\/$VERSION/d" $HOME/.bashrc
echo "Environment variables cleaned from .bashrc."

# Step 6: Reload shell
echo "Reloading shell configuration..."
source $HOME/.bashrc

# Step 7: Display success message
echo "mongosh $VERSION has been successfully uninstalled!"
