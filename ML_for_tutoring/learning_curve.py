def plot_learning_curve(estimator):
	plt.figure()
	plt.title("Learning SVM Curves")
	plt.ylim(0.7,1.01)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(estimator, images, labels)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
	plt.legend(loc="best")
	plt.show()
clf = svm.SVC(kernel='rbf',C=C_grid,gamma=gamma_grid)
#plot_learning_curve(clf)

def apply_pca(train_images,test_images,components):
	pca = PCA(n_components=components)
	train_images = pca.fit_transform(train_images)
	test_images = pca.transform(test_images)
	print train_images.shape
	print test_images.shape
	return train_images,test_images

train_images,test_images = apply_pca(train_images,test_images,0.7)
clf.fit(train_images, train_labels)
