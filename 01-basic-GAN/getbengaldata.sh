#clearing and creating the appropriate folders in the parent directory
#if [ -d "RawData" ]; then
#	rm -r "RawData"
#fi
#mkdir -p RawData
#if [ -d "Data" ]; then
#        rm -r "Data"
#fi
#mkdir -p Data
cd /projects/metoffice/ml-tc/ML-TC/RawData
#for each cyclone
for CYCLONE in BOB01 BOB07 TC01B AKASH SIDR RASHMI AILA VIYARU ROANU MORA FANI 
#for CYCLONE in AILA
do
	#check if the tarred file is there, and if not download it
	if ! test -f "../RawData/$CYCLONE.tar"; then
		echo "it doesn't exist"
		curl -# -o ../RawData/$CYCLONE.tar https://zenodo.org/api/files/476d7685-0a99-4f35-a32d-f6951e89dfec/tsens.$CYCLONE.tar.gz
	fi
	#then untar it
	#tar -zxvf "../RawData/"$CYCLONE".tar" -C  ../RawData
	#and for each variable run the data production script
	for VARIABLE in $2 
	do
		python3 /home/mo-txirouch/ML-TC/01-basic-GAN/making_data.py --f $1 --h $CYCLONE --v $VARIABLE --e $3 
	done
	#delete the tar file and its contents before downloading the next one to save space
	#rm -r "../RawData"
	#mkdir -p "../RawData"
done

