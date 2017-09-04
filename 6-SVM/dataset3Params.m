function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_ary = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_ary = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
best_error = 1e9;
C = C_ary(1);
sigma = sigma_ary(1);
for cur_C = C_ary
    for cur_sigma = sigma_ary
        model= svmTrain(X, y, cur_C, ...
            @(x1, x2) gaussianKernel(x1, x2, cur_sigma));
        predictions = svmPredict(model, Xval);
        cur_error = mean(double(predictions ~= yval));
        fprintf('cur_C=%.3f, cur_sigma=%.3f, Cur_error=%.3f', cur_C,...
            cur_sigma, cur_error);
        if cur_error < best_error
            best_error = cur_error;
            C = cur_C;
            sigma = cur_sigma;
        end
    end
end

fprintf('\nFinal C=%.3f, sigma=%.3f, best_error=%.3f',...
    C, sigma, best_error);




% =========================================================================

end
