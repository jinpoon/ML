data = [0.697,0.460; 0.774,0.376; 0.634,0.264; 0.608,0.318; 0.556,0.215; 0.403,0.237; 0.481,0.149
    0.437,0.211; 0.666,0.091; 0.243,0.267; 0.245,0.057; 0.343,0.099; 0.639,0.161; 0.657,0.198
    0.360,0.370; 0.593,0.042; 0.719,0.103];
label = [1;1;1;1;1;1;1;1;0;0;0;0;0;0;0;0;0];

ss = size(data,1);
trData = data([[1:5], [14:17]], :);
trLabel = label([[1:5], [14:17]], :);
trSet = [trData ones(size(trData, 1), 1)];

syms w1 w2 b;
w = [w1;w2;b];
f = 0;
for i = 1:size(trSet,1)
    xi = trSet(i,:);
    yi = trSet(i, 3);
    f = f +  (-yi)*xi*w + log(1+ exp(xi*w));   
end

df = [diff(f, 'w1'); diff(f, 'w2'); diff(f, 'b')];
df = conj(df');

%Newton

x0 = [0; 0; 1];
N = 2000;
eps = 0.001;
oldnf = 0;
nf = 0;
for i = 1:N
    nf = eval(subs(f, {'w1','w2', 'b'}, {x0(1), x0(2), x0(3)}));
    
    pl = zeros(size(trData, 1), 1);
    dl = 0;
    d2l = 0;
    
    for i = 1:size(trSet, 1)
        pl(i) = 1-1/(1+exp(trSet(i,:)*x0));
        dl = dl - trSet(i, :)*(trLabel(i)-pl(i));
        d2l = d2l + trSet(i,:) * trSet(i,:)' * pl(i) * (1-pl(i));
    end
    
    
    x0 = x0 - (d2l\dl)';
    if abs(oldnf-nf) < eps
        fprintf('%d iteration: Converge!\n', i);
        break;
    end
    oldnf = nf;
end
figure(1);
hold on;
line([0, -x0(3)/x0(2)], [-x0(3)/x0(1), 0]);
for i = 1:size(data,1)
    if label(i) == 1
        plot(data(i, 1), data(i, 2), 'r+');
    else
        plot(data(i, 1), data(i, 2), 'b+');
    end
end




