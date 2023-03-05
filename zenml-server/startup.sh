docker run -it -d -p 8080:8080 --name zenml \
    --mount type=bind,source=$PWD/zenml-server,target=/zenml/.zenconfig/local_stores/default_zen_store \
    zenmldocker/zenml-server
    