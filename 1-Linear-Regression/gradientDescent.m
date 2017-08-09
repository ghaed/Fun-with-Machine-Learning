function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    cur_theta = theta;
    h = X * theta;
    cur_sum_theta0 = 0;
    cur_sum_theta1 = 0;
    for i = 1:m 
        cur_sum_theta0 = cur_sum_theta0 + (h(i)-y(i))*X(i,1);
        cur_sum_theta1 = cur_sum_theta1 + (h(i)-y(i))*X(i,2);
    end
    cur_theta(1) = cur_theta(1) - alpha/m*cur_sum_theta0;
    cur_theta(2) = cur_theta(2) - alpha/m*cur_sum_theta1;
    theta = cur_theta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
