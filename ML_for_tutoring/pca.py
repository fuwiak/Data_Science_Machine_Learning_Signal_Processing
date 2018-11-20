def apply_pca(train_images,test_images,components):
	pca = PCA(n_components=components)
	train_images = pca.fit_transform(train_images)
	test_images = pca.transform(test_images)
	print train_images.shape
	print test_images.shape
	return train_images,test_images

train_images,test_images = apply_pca(train_images,test_images,0.7)
clf.fit(train_images, train_labels)
