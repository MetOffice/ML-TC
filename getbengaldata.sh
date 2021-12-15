for CYCLONE in BOB01 BOB07 TC01B AKASH SIDR RASHMI AILA VIYARU ROANU MORA FANI 
do
if ! test -f "$CYCLONE.tar"; then
    curl -# -o ../Data/$CYCLONE.tar https://zenodo.org/api/files/476d7685-0a99-4f35-a32d-f6951e89dfec/tsens.$CYCLONE.tar.gz
fi
done