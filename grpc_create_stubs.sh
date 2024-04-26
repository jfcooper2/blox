PATH_DEPLOYMENT="./blox/deployment"
PATH_PROTO=${PATH_DEPLOYMENT}/grpc_proto
PATH_STUBS=${PATH_DEPLOYMENT}/grpc_stubs
mkdir ${PATH_STUBS}

echo $PATH_PROTO
echo $PATH_PROTO
echo $PATH_STUBS

PROTOS=("backend" "frontend" "nm" "rm" "simulator")

for i in "${PROTOS[@]}"
do
    python3 -m grpc_tools.protoc --proto_path ${PATH_PROTO}/ --python_out=${PATH_STUBS} --pyi_out=${PATH_STUBS} --grpc_python_out=${PATH_STUBS} ${PATH_PROTO}/${i}.proto
done