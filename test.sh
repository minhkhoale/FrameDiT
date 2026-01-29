PASSED='results/taichi128/045-MatLatte-M-64-256-2-F16S3-taichi128-novae/checkpoints/1000000.pt'
if [[ -d $PASSED ]]; then
    echo "$PASSED is a directory"
elif [[ -f $PASSED ]]; then
    echo "$PASSED is a file"
else
    echo "$PASSED is not valid"
    exit 1
fi


if [[ ! -d $PASSED ]]; then
    echo "Local path does not exist: $PASSED"
fi