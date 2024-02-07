set -e
echo "build simple graph"
python examples/mywarmupmodule.py

echo "====================="
python extract_op_type_name.py
