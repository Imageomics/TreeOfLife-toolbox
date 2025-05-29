#!/bin/bash
BASE_DIR=""  # Set your base path for data
LOG_FILE="$BASE_DIR/mongodb.log"
DB_PATH="$BASE_DIR/mongodb_data"

echo "Starting MongoDB..."
mongod --dbpath=$DB_PATH --fork --logpath=$LOG_FILE
echo "MongoDB started!"

# mongod --dbpath="${BASE_DIR}/TreeOfLife/gbif_mongo" --fork --logpath="${BASE_DIR}/mongo_logs/gbif_mongo.log"
# Check process status
# ps aux | grep mongod

# Monitor MongoDB
# tail -f ${BASE_DIR}/mongodb.log

# Stop MongoDB
# mongod --shutdown --dbpath=${BASE_DIR}/mongodb_data


# Anther good option is to use tmux
# tmux new -s mongodb
# mongod --dbpath=${BASE_DIR}/mongodb_data

# Detach the session: Press Ctrl+B, then D.
# Reattach later if needed:
# tmux attach -t mongodb

# Note:
# `mongod` stands for "Mongo Daemon", which is the core process that runs the MongoDB database server.
# Mongo Daemon is responsible for 
# - Listening for client connections on a specified port (default: 27017)
# - Managing the storage, retrieval, and querying of data.
# - Performing database operations like replication and sharding.
# 
# `mongosh` / `mongo` are the client shell for interacting with the database
# `mongosh` needs to be installed separately 