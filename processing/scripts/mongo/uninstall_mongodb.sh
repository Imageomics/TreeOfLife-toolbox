#!/bin/bash

# Define variables
SOFTWARE_NAME="mongodb"
VERSION="8.0.4"
INSTALL_DIR="$HOME/local/$SOFTWARE_NAME/$VERSION"
MODULEFILES_DIR="$HOME/local/share/lmodfiles/$SOFTWARE_NAME"
MODULE_FILE="$MODULEFILES_DIR/$VERSION.lua"
BASE_DIR=""  # Set your base path for data
DATA_DIR="$BASE_DIR/mongodb_data"  # Custom data directory

# Step 1: Stop any running MongoDB server
echo "Stopping any running MongoDB servers..."
pkill -f "mongod --dbpath=$DATA_DIR" 2>/dev/null
if [ $? -eq 0 ]; then
  echo "MongoDB server stopped successfully."
else
  echo "No running MongoDB server found or could not stop the server."
fi

# Step 2: Remove MongoDB installation files
echo "Removing MongoDB installation directory..."
if [ -d "$INSTALL_DIR" ]; then
  rm -rf "$INSTALL_DIR"
  echo "MongoDB installation directory removed: $INSTALL_DIR"
else
  echo "MongoDB installation directory not found: $INSTALL_DIR"
fi

# Step 3: Remove the module file
echo "Removing MongoDB module file..."
if [ -f "$MODULE_FILE" ]; then
  rm -f "$MODULE_FILE"
  echo "MongoDB module file removed: $MODULE_FILE"
else
  echo "MongoDB module file not found: $MODULE_FILE"
fi

# Step 4: Remove the module directory if empty
if [ -d "$MODULEFILES_DIR" ] && [ -z "$(ls -A $MODULEFILES_DIR)" ]; then
  rmdir "$MODULEFILES_DIR"
  echo "Empty module directory removed: $MODULEFILES_DIR"
fi

# Step 5: Remove the data directory (optional)
read -p "Do you want to delete the data directory at $DATA_DIR? (y/n): " DELETE_DATA
if [ "$DELETE_DATA" == "y" ]; then
  if [ -d "$DATA_DIR" ]; then
    rm -rf "$DATA_DIR"
    echo "MongoDB data directory removed: $DATA_DIR"
  else
    echo "MongoDB data directory not found: $DATA_DIR"
  fi
else
  echo "MongoDB data directory retained: $DATA_DIR"
fi

# Step 6: Clean up environment variables in .bashrc
echo "Cleaning up environment variables in .bashrc..."
sed -i '/module use $HOME\/local\/share\/lmodfiles/d' $HOME/.bashrc
sed -i "/module load $SOFTWARE_NAME\/$VERSION/d" $HOME/.bashrc
echo "Environment variables cleaned from .bashrc."

# Step 7: Reload shell
echo "Reloading shell configuration..."
source $HOME/.bashrc

# Step 8: Display success message
echo "MongoDB $VERSION uninstalled successfully!"
