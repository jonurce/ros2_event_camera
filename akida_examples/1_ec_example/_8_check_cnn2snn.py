import cnn2snn

print("Checking cnn2snn Akida Version")
current_version = cnn2snn.get_akida_version()
print(f'Current version: {current_version}')
cnn2snn.set_akida_version(cnn2snn.AkidaVersion.v2)
updated_version = cnn2snn.get_akida_version()
print(f'Updated version: {updated_version}')