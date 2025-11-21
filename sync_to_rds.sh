#!/bin/bash
# ============================================
# Sync listed files to RDS using rsync
# ============================================

# ==== CONFIGURATION ====
RDS_USER="s224075134"
RDS_HOST="rds-storage.deakin.edu.au"
RDS_ROOT="/RDS/RDS75807-temporal-diffusion/storage2/projects/temporal_diffusion/video-diffusion-model-v2"
FILE_LIST="sync/file_to_sync.txt"   # list of local relative paths

# ==== MAIN LOOP ====
while IFS= read -r FILE_PATH; do
    # skip empty lines or comments
    # [[ -z "$FILE_PATH" || "$FILE_PATH" =~ ^# ]] && continue
    # get the directory path relative to file
    DIR_PATH=$(dirname "$FILE_PATH")

    echo "-----------------------------------------------------"
    echo -e "\tSyncing: $FILE_PATH"

    REMOTE_PATH="${RDS_ROOT}/${FILE_PATH}"
    REMOTE_DIR=$(dirname "$REMOTE_PATH")

    echo -e "\tChecking remote: $REMOTE_PATH"

    # check if remote file already exists
    # if ssh -n ${RDS_USER}@${RDS_HOST} "[ -f '$REMOTE_PATH' ]"; then
    #     echo "✅ File already exists on remote, skipping: $FILE_PATH"
    #     continue
    # fi

    # create remote directory if it doesn't exist
    ssh -n ${RDS_USER}@${RDS_HOST} "mkdir -p '$RDS_ROOT/$DIR_PATH'"

    if [[ -d "$FILE_PATH" ]]; then
        echo -e "\t📁 Incrementally syncing directory: $FILE_PATH → $REMOTE_PATH"
        rsync -a --info=progress "$FILE_PATH"/ "${RDS_USER}@${RDS_HOST}:${REMOTE_PATH}/"

    elif [[ -f "$FILE_PATH" ]]; then
        echo -e "\t📄 Syncing single file"
        if ssh -n ${RDS_USER}@${RDS_HOST} "[ -f '$REMOTE_PATH' ]"; then
            echo -e "\t✅ File exists, skipping: $FILE_PATH"
        else
            rsync -a --info=progress "$FILE_PATH" "${RDS_USER}@${RDS_HOST}:${REMOTE_PATH}"
        fi
    else
        echo -e "\t⚠️ Unknown type (not file or folder): $FILE_PATH"
    fi
    #rsync -a --info=progress "$FILE_PATH" "${RDS_USER}@${RDS_HOST}:${RDS_ROOT}/$FILE_PATH"

done < "$FILE_LIST"

echo "✅ All files synced successfully."
