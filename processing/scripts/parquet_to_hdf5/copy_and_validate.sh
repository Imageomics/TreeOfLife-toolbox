#!/bin/bash

# TreeOfLife Copy and Validate Script
# Copies converted HDF5/Parquet files from source to destination storage with validation

# Function to display usage
usage() {
    cat << EOF
Usage: $0 --source-root SOURCE_ROOT --dest-root DEST_ROOT [--email EMAIL]

Required arguments:
    --source-root    Source directory including source= prefix (e.g., /path/to/data/source=gbif_group_20)
    --dest-root      Destination directory including source= prefix (e.g., /path/to/data/source=gbif)

Optional arguments:
    --email          Email address for success notification

EOF
}

# Parse command line arguments
SOURCE_ROOT=""
DEST_ROOT=""
EMAIL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --source-root)
            SOURCE_ROOT="$2"
            shift 2
            ;;
        --dest-root)
            DEST_ROOT="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SOURCE_ROOT" || -z "$DEST_ROOT" ]]; then
    echo "Error: Missing required arguments"
    echo ""
    usage
    exit 1
fi

# Use the full paths directly (no building required)
SOURCE_BASE="$SOURCE_ROOT"
DEST_BASE="$DEST_ROOT"

# Functions for verification
verify_sizes() {
    local source_base=$1
    local dest_base=$2
    local all_match=true
    
    echo "Running size verification..."
    for server in "$source_base"/server=*; do
        server_name=$(basename "$server")
        echo "  Checking sizes for $server_name..."
        source_hash=$( cd "$server" && du -b *_metadata.parquet *_images.h5 2>/dev/null | sort | md5sum )
        dest_hash=$( cd "$dest_base/$server_name" && du -b *_metadata.parquet *_images.h5 2>/dev/null | sort | md5sum )
        if [ "$source_hash" = "$dest_hash" ]; then
            echo "    ✓ sizes match"
        else
            echo "    ✗ size MISMATCH!"
            echo "      Source: $source_hash"
            echo "      Dest:   $dest_hash"
            all_match=false
        fi
    done
    
    [ "$all_match" = true ]
}

verify_checksums() {
    local source_base=$1
    local dest_base=$2
    local all_match=true
    
    echo "Running checksum verification..."
    for server in "$source_base"/server=*; do
        server_name=$(basename "$server")
        echo "  Checking checksums for $server_name..."
        source_hash=$( cd "$server" && md5sum *_metadata.parquet *_images.h5 2>/dev/null | sort | md5sum )
        dest_hash=$( cd "$dest_base/$server_name" && md5sum *_metadata.parquet *_images.h5 2>/dev/null | sort | md5sum )
        if [ "$source_hash" = "$dest_hash" ]; then
            echo "    ✓ checksums match"
        else
            echo "    ✗ checksum MISMATCH!"
            echo "      Source: $source_hash"
            echo "      Dest:   $dest_hash"
            all_match=false
        fi
    done
    
    [ "$all_match" = true ]
}

# Function to send email notification
send_success_email() {
    local source_base=$1
    local dest_base=$2
    local start_time=$3
    local end_time=$4
    local email_addr=$5

    if [[ -z "$email_addr" ]]; then
        return 0
    fi

    # Calculate execution time
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    # Count files and calculate total size
    local file_count=0
    local total_size=0

    for server in "$source_base"/server=*; do
        if [[ -d "$server" ]]; then
            local server_files=$(find "$server" -name "*_metadata.parquet" -o -name "*_images.h5" | wc -l)
            local server_size=$(find "$server" -name "*_metadata.parquet" -o -name "*_images.h5" -exec stat -c%s {} + | awk '{sum+=$1} END {print sum+0}')
            file_count=$((file_count + server_files))
            total_size=$((total_size + server_size))
        fi
    done

    # Convert size to human readable format
    local size_gb=$((total_size / 1024 / 1024 / 1024))
    local size_mb=$(((total_size % (1024 * 1024 * 1024)) / 1024 / 1024))

    # Extract source group name from path for display
    local source_group=$(basename "$source_base")

    # Create email content
    local subject="TreeOfLife Copy Success: $source_group"
    local body="TreeOfLife Data Copy and Validation - SUCCESS

==========================================
OPERATION SUMMARY
==========================================
Status: SUCCESS - All files copied and verified
Source Group: $source_group
Execution Time: ${hours}h ${minutes}m ${seconds}s
Completed: $(date)

==========================================
DATA DETAILS
==========================================
Source Path: $source_base
Destination: $dest_base
Files Copied: $file_count
Total Size: ${size_gb}.$(printf "%03d" $((size_mb * 1000 / 1024)))GB

==========================================
VERIFICATION RESULTS
==========================================
✓ Size verification: PASSED
✓ MD5 checksum verification: PASSED
✓ All files successfully validated

==========================================
NEXT STEPS
==========================================
Data has been successfully copied to research storage.
Scratch files can now be safely deleted if desired.

Job completed on: $(hostname)
SLURM Job ID: ${SLURM_JOB_ID:-"N/A (manual execution)"}
"

    # Send email using mail command
    echo "$body" | mail -s "$subject" "$email_addr"
    echo "Success notification sent to: $email_addr"
}

# Main workflow
START_TIME=$(date +%s)
echo "============================================="
echo "TREE OF LIFE DATA COPY AND VERIFICATION"
echo "============================================="
echo "Source:      $SOURCE_BASE"
echo "Destination: $DEST_BASE"
echo ""

# Step 1: Validate destination directories exist
echo "=== Step 1: Validating destination directories ==="
for server in "$SOURCE_BASE"/server=*; do
    server_name=$(basename "$server")
    if [ ! -d "$DEST_BASE/$server_name" ]; then
        echo "ERROR: $server_name missing in destination"
        exit 1
    fi
done
echo "✓ All destination directories exist"
echo ""

# Step 2: Copy files
echo "=== Step 2: Copying files ==="
for server in "$SOURCE_BASE"/server=*; do
    server_name=$(basename "$server")
    echo "Copying $server_name..."
    cp -v "$server"/*_metadata.parquet "$server"/*_images.h5 "$DEST_BASE/$server_name/" || {
        echo "ERROR: Copy failed for $server_name"
        exit 1
    }
done
echo "✓ Copy complete"
echo ""

# Step 3: Size verification
echo "=== Step 3: Size verification (fast) ==="
if verify_sizes "$SOURCE_BASE" "$DEST_BASE"; then
    echo "✓ All size checks passed"
    echo ""
else
    echo "✗ Size verification FAILED. Stopping."
    exit 1
fi

# Step 4: Checksum verification
echo "=== Step 4: Checksum verification (slow) ==="
if verify_checksums "$SOURCE_BASE" "$DEST_BASE"; then
    echo "✓ All checksum verifications passed"
    echo ""

    # Calculate end time and send email notification
    END_TIME=$(date +%s)

    echo "============================================="
    echo "SUCCESS: All files copied and verified"
    echo "============================================="

    # Send success email if email address provided
    if [[ -n "$EMAIL" ]]; then
        echo "Sending success notification..."
        send_success_email "$SOURCE_BASE" "$DEST_BASE" "$START_TIME" "$END_TIME" "$EMAIL"
    fi

    echo "Copy and validation completed successfully!"
else
    echo "✗ Checksum verification FAILED"
    exit 1
fi
