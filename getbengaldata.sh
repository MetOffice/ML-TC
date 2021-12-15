mkdir -p ../RawData
for CYCLONE in BOB01 BOB07 TC01B AKASH SIDR RASHMI AILA VIYARU ROANU MORA FANI 
do
if ! test -f "../RawData/$CYCLONE.tar"; then
echo "it doesn't exist"
curl -# -o ../RawData/$CYCLONE.tar https://zenodo.org/api/files/476d7685-0a99-4f35-a32d-f6951e89dfec/tsens.$CYCLONE.tar.gz
fi
done
