"""plsda
=======
Partial Least Squares Discriminant Analysis (PLS-DA).

Small scikit-learn-compatible classifier built on top of
``sklearn.cross_decomposition.PLSRegression``. The estimator fits PLS on a
one-hot/dummy-coded target matrix and maps the resulting continuous class
scores back to labels using one of five classification rules.

The public API is a single class:

* :class:`PLSDA` -- sklearn estimator supporting ``fit``, ``predict_scores``,
  ``decision_function``, ``predict``, ``explained_variance_x`` and
  ``explained_variance_y``.

Classification Rules
--------------------
``argmax``
    Predicts the class whose dummy-target column has the largest predicted
    PLS response. This is the default and cheapest rule.

``centroids``
    Projects samples into the PLS X-score space and predicts the closest class
    centroid using squared Euclidean distance.

``mahalanobis``
    Uses the same class centroids, but distances are weighted by the pooled
    within-class residual covariance matrix in the PLS X-score space.

``lda``
    Fits a Linear Discriminant Analysis classifier on the PLS X-scores and
    applies it to the scores of new samples.

``qda``
    Fits a Quadratic Discriminant Analysis classifier on the PLS X-scores and
    applies it to the scores of new samples.

scikit-learn Contract
---------------------
The constructor stores parameters without side effects, so ``get_params``,
``set_params``, ``clone`` and ``GridSearchCV`` work through ``BaseEstimator``.
Learned values are exposed with trailing underscores (``classes_``, ``pls_``,
``centroids_``, ``inv_covariance_``, ``lda_``, ``qda_``), following scikit-learn
conventions.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class PLSDA(ClassifierMixin, BaseEstimator):
    """
    Partial Least Squares Discriminant Analysis classifier.

    The model fits :class:`sklearn.cross_decomposition.PLSRegression` on a
    dummy-coded target matrix with one column per class. Predictions are then
    decoded using one of five classification rules:

    * ``"argmax"``: largest predicted dummy-target response;
    * ``"centroids"``: nearest class centroid in PLS X-score space;
    * ``"mahalanobis"``: covariance-weighted nearest centroid;
    * ``"lda"``: Linear Discriminant Analysis fitted on PLS X-scores.
    * ``"qda"``: Quadratic Discriminant Analysis fitted on PLS X-scores.

    Parameters
    ----------
    n_components : int, default=2
        Number of latent variables passed to ``PLSRegression``.

    classification_method : {"argmax", "centroids", "mahalanobis", "lda", "qda"}, default="argmax"
        Rule used by :meth:`predict` to convert the fitted PLS representation
        into class labels.

    scale : bool, default=False
        Whether ``PLSRegression`` should scale ``X`` and the dummy-coded
        target matrix before fitting.

    max_iter : int, default=500
        Maximum number of iterations used by the NIPALS algorithm inside
        ``PLSRegression``.

    tol : float, default=1e-6
        Convergence tolerance passed to ``PLSRegression``.

    copy : bool, default=True
        Whether ``PLSRegression`` copies ``X`` and ``y`` during fitting.

    lda_solver : {"svd", "lsqr", "eigen"}, default="svd"
        Solver used by the LDA model when
        ``classification_method="lda"``.

    lda_shrinkage : "auto", float or None, default=None
        Shrinkage parameter used by LDA. Shrinkage is supported only by the
        ``"lsqr"`` and ``"eigen"`` solvers.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during fitting, ordered as returned by
        :func:`numpy.unique`.

    pls_ : PLSRegression
        Fitted underlying PLS regression model.

    centroids_ : ndarray of shape (n_classes, n_components)
        Class centroids in PLS X-score space. Available only for
        ``classification_method`` equal to ``"centroids"`` or
        ``"mahalanobis"``.

    inv_covariance_ : ndarray of shape (n_components, n_components)
        Pseudoinverse of the pooled within-class residual covariance matrix.
        Available only for ``classification_method="mahalanobis"``.

    lda_ : LinearDiscriminantAnalysis
        LDA model fitted on the PLS X-scores. Available only for
        ``classification_method="lda"``.
    
    qda_ : QuadraticDiscriminantAnalysis
        QDA model fitted on the PLS X-scores. Available only for
        ``classification_method="qda"``.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during :meth:`fit`, when ``X`` has string column
        names.

    explained_variance_x_ : ndarray of shape (n_components,)
        Incremental fraction of the centered, and optionally scaled, ``X``
        sum of squares explained by each PLS component.

    explained_variance_y_ : ndarray of shape (n_components,)
        Incremental fraction of the centered, and optionally scaled,
        dummy-coded target sum of squares explained by each PLS component.

    Examples
    --------
    Basic classifier using the argmax rule::

        model = PLSDA(n_components=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    PLS followed by LDA::

        model = PLSDA(
            n_components=3,
            classification_method="lda",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    Grid search over latent variables::

        from sklearn.model_selection import GridSearchCV

        search = GridSearchCV(
            PLSDA(classification_method="centroids"),
            {"n_components": [1, 2, 3]},
            cv=5,
        )
        search.fit(X_train, y_train)
    """

    _VALID_METHODS = {
        "argmax",
        "centroids",
        "mahalanobis",
        "lda",
        "qda"
    }

    def __init__(
        self,
        n_components=2,
        classification_method="argmax",
        scale=False,
        max_iter=500,
        tol=1e-6,
        copy=True,
        lda_solver="svd",
        lda_shrinkage=None,
    ):
        self.n_components = n_components
        self.classification_method = classification_method
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.lda_solver = lda_solver
        self.lda_shrinkage = lda_shrinkage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the PLS-DA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.

        y : array-like of shape (n_samples,)
            Class labels. At least two distinct classes are required.

        Returns
        -------
        self : PLSDA
            Fitted estimator.
        """
        X, y = validate_data(self, X, y, reset=True)
        check_classification_targets(y)
        self._check_classification_method()
        self._check_lda_parameters()

        for attribute in (
            "centroids_",
            "inv_covariance_",
            "lda_",
            "qda_",
            "explained_variance_x_",
            "explained_variance_y_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)

        self.classes_, y_indices = np.unique(y, return_inverse=True)

        if self.classes_.size < 2:
            raise ValueError("PLSDA needs at least two classes; received 1 class.")

        # PLSRegression is a regressor, so class labels are converted into a
        # dummy-coded target matrix with one column per class.
        y_dummy = np.eye(self.classes_.size, dtype=float)[y_indices]
        X_reference = np.asarray(X, dtype=float).copy()
        y_reference = y_dummy.copy()

        self.pls_ = PLSRegression(
            n_components=self.n_components,
            scale=self.scale,
            max_iter=self.max_iter,
            tol=self.tol,
            copy=self.copy,
        )
        self.pls_.fit(X, y_dummy)
        self.n_iter_ = np.asarray(self.pls_.n_iter_)
        self._fit_explained_variance(X_reference, y_reference)

        # The PLS model is common to every classification method. Additional
        # parameters are fitted only when required by the selected rule.
        if self.classification_method == "centroids":
            self._fit_centroids(y_indices)

        elif self.classification_method == "mahalanobis":
            self._fit_centroids(y_indices)
            self._fit_mahalanobis(y_indices)

        elif self.classification_method == "lda":
            self._fit_lda(y_indices)

        elif self.classification_method == "qda":
            self._fit_qda(y_indices)

        return self

    def predict_scores(self, X):
        """
        Return the continuous PLS dummy-target predictions.

        These are the direct outputs of ``PLSRegression.predict`` before any
        classification rule is applied.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Predicted dummy-target responses.
        """
        check_is_fitted(self, "pls_")
        X = validate_data(self, X, reset=False)
        return np.asarray(self.pls_.predict(X))

    def explained_variance_x(self):
        """Return the incremental fraction of X variance per component.

        The fractions refer to the centered training matrix. When
        ``scale=True``, they refer to the centered and standardized matrix.

        Returns
        -------
        ndarray of shape (n_components,)
            Fraction explained by each PLS component. The returned array is a
            copy and can be modified without changing the fitted estimator.
        """
        check_is_fitted(self, "explained_variance_x_")
        return self.explained_variance_x_.copy()

    def explained_variance_y(self):
        """Return the incremental fraction of dummy-target variance per component.

        The fractions refer to the centered dummy-coded training target. When
        ``scale=True``, they refer to the centered and standardized target.

        Returns
        -------
        ndarray of shape (n_components,)
            Fraction explained by each PLS component. The returned array is a
            copy and can be modified without changing the fitted estimator.
        """
        check_is_fitted(self, "explained_variance_y_")
        return self.explained_variance_y_.copy()

    def decision_function(self, X):
        """
        Return classification scores for the selected decision rule.

        Higher values indicate stronger support for a class. For binary
        classification, a one-dimensional score is returned, following the
        scikit-learn classifier convention. For multiclass classification,
        one score per class is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Classification scores associated with the configured method.
        """
        check_is_fitted(self, ["pls_", "classes_"])
        X = validate_data(self, X, reset=False)

        if self.classification_method == "argmax":
            scores = np.asarray(self.pls_.predict(X))

        elif self.classification_method == "centroids":
            scores = -self._centroid_distances(X)

        elif self.classification_method == "mahalanobis":
            scores = -self._mahalanobis_distances(X)

        elif self.classification_method == "lda":
            check_is_fitted(self, "lda_")
            x_scores = np.asarray(self.pls_.transform(X))
            return np.asarray(self.lda_.decision_function(x_scores))

        elif self.classification_method == "qda":
            check_is_fitted(self, "qda_")
            x_scores = np.asarray(self.pls_.transform(X))
            return np.asarray(self.qda_.decision_function(x_scores))

        else:
            self._check_classification_method()
            raise RuntimeError("Unreachable classification method state.")

        if self.classes_.size == 2:
            return scores[:, 1] - scores[:, 0]

        return scores

    def predict(self, X):
        """
        Predict class labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["pls_", "classes_"])
        X = validate_data(self, X, reset=False)

        # Keep the dispatch explicit: every classification method has a
        # clearly separated private prediction function in this same file.
        if self.classification_method == "argmax":
            class_indices = self._predict_argmax(X)

        elif self.classification_method == "centroids":
            class_indices = self._predict_centroids(X)

        elif self.classification_method == "mahalanobis":
            class_indices = self._predict_mahalanobis(X)

        elif self.classification_method == "lda":
            class_indices = self._predict_lda(X)

        elif self.classification_method == "qda":
            class_indices = self._predict_qda(X)

        else:
            # This normally cannot happen after a successful fit, but it also
            # protects against changing the parameter manually after fitting.
            self._check_classification_method()
            raise RuntimeError("Unreachable classification method state.")

        return self.classes_[class_indices]

    # ------------------------------------------------------------------
    # Argmax classification
    # ------------------------------------------------------------------

    def _predict_argmax(self, X):
        """Return class indices from the largest predicted dummy response."""
        y_scores = np.asarray(self.pls_.predict(X))
        return np.argmax(y_scores, axis=1)

    # ------------------------------------------------------------------
    # Centroid classification
    # ------------------------------------------------------------------

    def _fit_centroids(self, y_indices):
        """Store one class centroid in the fitted PLS X-score space."""
        x_scores = np.asarray(self.pls_.x_scores_)

        self.centroids_ = np.vstack(
            [
                x_scores[y_indices == class_index].mean(axis=0)
                for class_index in range(self.classes_.size)
            ]
        )

    def _centroid_distances(self, X):
        """Return squared Euclidean distances to every class centroid."""
        check_is_fitted(self, ["pls_", "centroids_"])

        x_scores = np.asarray(self.pls_.transform(X))

        # Shape of delta:
        # (n_samples, n_classes, n_components)
        delta = x_scores[:, None, :] - self.centroids_[None, :, :]

        return np.sum(delta**2, axis=2)

    def _predict_centroids(self, X):
        """Return nearest-centroid class indices."""
        distances = self._centroid_distances(X)
        return np.argmin(distances, axis=1)

    # ------------------------------------------------------------------
    # Mahalanobis classification
    # ------------------------------------------------------------------

    def _fit_mahalanobis(self, y_indices):
        """
        Store the inverse pooled within-class covariance in score space.

        Residuals are computed from each training score to the centroid of its
        observed class. ``pinv`` is used instead of ``inv`` so the classifier
        remains usable when the covariance matrix is singular or nearly
        singular.
        """
        x_scores = np.asarray(self.pls_.x_scores_)
        residuals = x_scores - self.centroids_[y_indices]

        covariance = residuals.T @ residuals / residuals.shape[0]
        covariance = np.atleast_2d(covariance)

        self.inv_covariance_ = np.linalg.pinv(covariance)

    def _mahalanobis_distances(self, X):
        """Return squared Mahalanobis distances to every class centroid."""
        check_is_fitted(
            self,
            ["pls_", "centroids_", "inv_covariance_"],
        )

        x_scores = np.asarray(self.pls_.transform(X))

        # Shape of delta:
        # (n_samples, n_classes, n_components)
        delta = x_scores[:, None, :] - self.centroids_[None, :, :]

        # For every sample i and class k, compute:
        # (t_i - centroid_k)^T @ inv_covariance @ (t_i - centroid_k)
        return np.einsum(
            "nki,ij,nkj->nk",
            delta,
            self.inv_covariance_,
            delta,
        )

    def _predict_mahalanobis(self, X):
        """Return covariance-weighted nearest-centroid class indices."""
        distances = self._mahalanobis_distances(X)
        return np.argmin(distances, axis=1)

    # ------------------------------------------------------------------
    # PLS-LDA classification
    # ------------------------------------------------------------------

    def _fit_lda(self, y_indices):
        """Fit Linear Discriminant Analysis on the training PLS X-scores."""
        self.lda_ = LDA(
            solver=self.lda_solver,
            shrinkage=self.lda_shrinkage,
        )
        self.lda_.fit(self.pls_.x_scores_, y_indices)

    def _predict_lda(self, X):
        """Return class indices predicted by LDA in PLS X-score space."""
        check_is_fitted(self, ["pls_", "lda_"])
        x_scores = np.asarray(self.pls_.transform(X))
        return self.lda_.predict(x_scores)


    # ------------------------------------------------------------------
    # PLS-QDA classification
    # ------------------------------------------------------------------
    def _fit_qda(self, y_indices):
        """Fit Quadratic Discriminant Analysis on the training PLS X-scores."""
        self.qda_ = QDA()
        self.qda_.fit(self.pls_.x_scores_, y_indices)

    def _predict_qda(self, X):
        """Return class indices predicted by QDA in PLS X-score space."""
        check_is_fitted(self, ["pls_", "qda_"])
        x_scores = np.asarray(self.pls_.transform(X))
        return self.qda_.predict(x_scores)

    # ------------------------------------------------------------------
    # Explained variance
    # ------------------------------------------------------------------

    def _fit_explained_variance(self, X, y_dummy):
        """Store per-component explained sums of squares for X and dummy Y."""
        X -= X.mean(axis=0)
        y_dummy -= y_dummy.mean(axis=0)

        if self.scale:
            X_std = X.std(axis=0, ddof=1)
            X_std[X_std == 0] = 1
            X /= X_std

            y_std = y_dummy.std(axis=0, ddof=1)
            y_std[y_std == 0] = 1
            y_dummy /= y_std

        self.explained_variance_x_ = self._explained_variance_ratios(
            X,
            self.pls_.x_scores_,
            self.pls_.x_loadings_,
        )
        self.explained_variance_y_ = self._explained_variance_ratios(
            y_dummy,
            self.pls_.x_scores_,
            self.pls_.y_loadings_,
        )

    @staticmethod
    def _explained_variance_ratios(matrix, scores, loadings):
        """Return the residual sum-of-squares reduction of each component."""
        residual = np.asarray(matrix, dtype=float).copy()
        total_sum_squares = np.sum(residual**2)
        ratios = np.zeros(scores.shape[1], dtype=float)

        if total_sum_squares == 0:
            return ratios

        previous_sum_squares = total_sum_squares
        for component in range(scores.shape[1]):
            residual -= np.outer(scores[:, component], loadings[:, component])
            current_sum_squares = np.sum(residual**2)
            ratios[component] = (
                previous_sum_squares - current_sum_squares
            ) / total_sum_squares
            previous_sum_squares = current_sum_squares

        return np.clip(ratios, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def _check_classification_method(self):
        """Validate the configured classification rule."""
        if self.classification_method not in self._VALID_METHODS:
            supported = ", ".join(
                repr(method) for method in sorted(self._VALID_METHODS)
            )
            raise ValueError(
                "Unknown classification_method="
                f"{self.classification_method!r}. "
                f"Supported values are: {supported}."
            )

    def _check_lda_parameters(self):
        """Validate parameters used by the optional LDA classification rule."""
        if self.classification_method != "lda":
            return

        valid_solvers = {"svd", "lsqr", "eigen"}
        if self.lda_solver not in valid_solvers:
            supported = ", ".join(repr(value) for value in sorted(valid_solvers))
            raise ValueError(
                f"Unknown lda_solver={self.lda_solver!r}. "
                f"Supported values are: {supported}."
            )

        if self.lda_solver == "svd" and self.lda_shrinkage is not None:
            raise ValueError(
                "lda_shrinkage is not supported when lda_solver='svd'. "
                "Use lda_solver='lsqr' or lda_solver='eigen', or set "
                "lda_shrinkage=None."
            )
