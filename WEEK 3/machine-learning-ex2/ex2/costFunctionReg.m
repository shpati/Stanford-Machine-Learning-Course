function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Recursive implementation

% STEP-1: Cost function J(theta) and Gradient calculations (un-regularized)
for i=1:m,
	J = J + (-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)))/m;
	grad = grad + (sigmoid(X(i,:)*theta)-y(i)).*X(i,:)'/m;
end;

% STEP-2: Cost function J(theta) and Gradient calculations (regularized)

for j=2:size(theta,1),
	J = J + lambda/2/m*theta(j)^2;
end;

grad(2:end) = grad(2:end) + lambda/m*theta(2:end);

% =============================================================

end
