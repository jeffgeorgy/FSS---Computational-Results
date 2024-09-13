tic
%% Limpa a memoria e carrega os dados
clear
clc
close all
% Carrega os dados da planta
dados = load('dim032.txt');
dados = dados';
centros = load('c-dim032.txt');
centros = centros';
qt_rotulos = length(centros(1,:));
dados_ = zeros(size(dados));
centros_ = zeros(size(centros));
for i = 1:length(dados(:,1))
    mx = max(dados(i,:));
    mn = min(dados(i,:));
    if mx == mn
        break
    else
        dados_(i,:) = (dados(i,:)-mn)/(mx-mn);
        centros_(i,:) = (centros(i,:)-mn)/(mx-mn);
    end
end
%% Inicializa as variaveis do Algoritmo ETS
% Inicializa a contagem de tempo 0.2
% 0.1 0.32 2.8 0.9 0.9
% Parâmetros do algoritmo
r = 1; % raio de influencia de um cluster
epsilon = 0.3; % Fator de raio de cluster
m = 3.2; %Ponderamento exponencial para calculo de pertinencias 2.8
gama = 0.9; % Ponderamento das informacoes no potencial 0.775
s = 0.9; % Fator de sensibilidade para determinacao de candidato 0.81
nmm = 1; % Tamanho da janela de media movel
% Variaveis do algoritmo
alpha = 4/r^2; % Parametro dependente de r
k = 1; % Amostra atual
R = 1; % Numero de regras inicial
xn = dados_; % Vetor de dados normalizados
x = dados; % Vetor de dados original
n = length(xn(:,1)); % Tamanho de dados de entrada
xcn{1} = [xn(:,1)]; % Primeiro centro de cluster
N = length(xn(1,:)); % Tamando dos dados
jan = zeros(2,nmm); % Janela de media movel
varphi_b = 0; % Valor inicial da taxa de variacao
tvn_c = 0; % Valor da taxa de variacao do primeiro centro
P(1) = 1; % Potencial do primeiro dado
Pc{1} = [P(1),0]; % Na primeira coluna, o potencial do centro, na segunda
% a quantidade de dados usados para medi-lo
iuc = 1; % indice do ultimo centro de cluster encontrado
vmax = 0; % Valor maximo da taxa de variacao para normalizacao
NF(:,:,1) = r*eye(n); % Inicializacao do numerador da matriz de covariancia
% fuzzy
DF(1) = 1; % Inicializacao do denominador da matriz de covariancia fuzzy
F(:,:,1) = NF(:,:,1)/DF(1);  % Inicializacao da matriz de covariancia fuzzy
D(1,1) = 0; % Distancia do primeiro dado ao primeiro centro
mu(1,1) = 1; % pertinencia do primeiro dado ao primeiro centro
pex = 1; % Peso exponencial no calcula da distancia adptativa
qt_reg = 1;

% figure % inicia uma nova figura
% plot(u_on,y_on,'k'); % Plota a curva estatica
% hold on % Segurar os graficos ja plotados
% plot(xcn{1}(1,:),xcn{1}(2,:),'k*') % Plota os centros normalizados

%% Realizacao do algoritmo
% O algoritmo agrupa apenas os dados x, sem considerar a saida y, e
% consequentemente o vetor inteiro de dados z
% plot(1,Pc{1},'ko')
% hold on
for k = 2:N
    k % Mostra o valor de k
    %Calcula a taxa de variacao
    txv(:,k) = xn(:,k) - xn(:,k-1); % Calcula a taxa de variacao da amostra
    %atual
    if nmm == 1
        txvf(:,k) = txv(:,k); % Se a janela for de 1 unidade, a taxa filtrada 
        % e igual a ela propria
    else
        jan = [jan(:,2:nmm),txv(:,k)]; % Atualiza a janela
        txvf(:,k) = mean(jan,2); % Calcula a taxa filtrada
    end
    varphi(k) = norm(txvf(:,k)); % Medicao de variacao pela norma
    if varphi(k) > vmax
        vmax = varphi(k); % Atualiza a medicao maxima se for o caso
    end
    if vmax ~= 0
        varphi_b(k) = varphi(k)/vmax; % Atualiza a medicao normalizada
    else 
        varphi_b(k) = varphi(k); % Atualiza a medicao normalizada
    end
    Pc{k,1} = Pc{k-1}; % Potencial de centros no instante atual
    xcn{k,1} = xcn{k-1}; % Centros no instante atual
    for i = 1:R % Medicao da distancia adaptativa do dado atual para os centros
        dist = sqrt((xn(:,k)-xcn{k}(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
            (xn(:,k)-xcn{k}(:,i)));
        D(i,k) = pex*(exp(dist)-1) + (1-pex)*dist;
    end
    P(k,1) = 0; % Inicializa o potencial da amostra atual
    Nda = k-iuc-1; % Atualiza a quantidade de dados usados na medicao do dado atual
    % Calcula o potencial do novo dado primeiro com a relação aos
    % centros existentes e atualiza os potenciais dos centros existentes
    for j = 1:R
        P(k,1) = P(k) + 1/(R+Nda)*((1-gama)*exp(-alpha*D(j,k)^2) + gama*exp(...
            -alpha*varphi_b(k)^2)); % Potencial do dado atual
        Pc{k}(j,2) = Pc{k}(j,2) + 1; % Numero de amostras usadas nos potenciais dos centros
        Pc{k}(j,1) = (Pc{k}(j,2)-1)/Pc{k}(j,2)*Pc{k}(j,1) + 1/Pc{k}(j,2)*...
            ((1-gama)*exp(-alpha*D(j,k)^2) + gama*exp(-alpha*tvn_c(j)^2)); % Potenciais
        % de centros
    end
    % Calcula o potencial do novo dado depois com a relação aos
    % dados pos ultimo cluster
    for j = (iuc+1):(k-1)
        P(k,1) = P(k) + 1/(R+Nda)*((1-gama)*exp(-alpha*norm(xn(:,k)-xn(:,j))^2)...
            + gama*exp(-alpha*varphi_b(k)^2)); %Potencial do dado atual
    end
    % Avalia as condicoes para evolucao da estrutura
    if P(k) > s*max(Pc{k}(:,1)) % Se o potencial do dado atual for maior
        novo_dado = 1;
        % que o fator de sensibilidade vezes o maior potencial de centro
        dmin = 1e6; % Inicializa a distancia
        for i = 1:R % Calcula a distancia para o centro mais proximo
            if D(i,k) < dmin
                dmin = D(i,k);
                indc_prox = i;
            end
        end
        if dmin < epsilon*r
            rou = P(k)/(P(k)+Pc{k}(indc_prox,1)); % Peso para o dado atual
            % Novas coordenadas de centro de cluster
            xcn{k}(:,indc_prox) = rou*xn(:,k) + (1-rou)*xcn{k}(:,indc_prox);
            % Atualiza o potencial e a quantidade de dados considerados
            % para a sua medição
            Pc{k}(indc_prox,:) = [rou*P(k)+(1-rou)*Pc{k}(indc_prox,1),ceil(rou*...
                (R+Nda)+(1-rou)*Pc{k}(indc_prox,2))];
            tvn_c(indc_prox) = rou*varphi_b(k) + (1-rou)*tvn_c(indc_prox);
            % Atualiza a distância do dado atual para o novo centro
            % resultante do curzamento
            dist = sqrt((xn(:,k)-xcn{k}(:,indc_prox))'*det(F(:,:,indc_prox))^(1/n)*inv(F(:,:,indc_prox))*...
            (xn(:,k)-xcn{k}(:,indc_prox)));
            D(indc_prox,k) = pex*(exp(dist)-1) + (1-pex)*dist;
            % Plota o novo centro
        else
            R = R + 1; % Aumenta o numero de regras
            iuc = k; % atualiza o indice do ultimo centro criado para o atual
            xcn{k}(:,R) = xn(:,k); % Centro da nova regra
            Pc{k}(R,1) = P(k); % Potencial do novo centro
            Pc{k}(R,2) = R-1 + Nda; % Numero de dados usados para medicao do potencial
            tvn_c(R,1) = varphi_b(k); % Taxa de variacao do centro atual
            % Inicializa a matriz de covariancia fuzzy quando se cria uma nova regra
            NF(:,:,R) = r*eye(n); % Numerador
            DF(R,1) = 1; % Denominador
            F(:,:,R) = NF(:,:,R)/DF(R); % Matriz de covariancia fuzzy
            % for j = 1:k
            %     for i = 1:R-1 % Medicao da distancia adaptativa do dado atual para os centros
            %         dist = sqrt((xn(:,j)-xcn{k}(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
            %             (xn(:,j)-xcn{k}(:,i)));
            %         D(i,j) = pex*(exp(dist)-1) + (1-pex)*dist;
            %     end
            %     % Armazena os indices de elementos nulos em D(:,k)
            %     indcn = find(D(:,j)==0);
            %     if isempty(indcn) % Se nao houver elementos nulos
            %         for i = 1:R-1
            %             mu(i,j) = 0;
            %             for jj = 1:R-1
            %                 mu(i,j) = mu(i,j) + (D(i,j)/D(jj,j))^(2/(m-1));
            %             end
            %             mu(i,j) = 1/mu(i,j);
            %         end
            %     else % Se houver elementos nulos em D(:,k)
            %         % Divide igualmente a pertinencia nos clusters em que D(i,k)=0,
            %         % e deixa pertinencia 0 nos demais clusters
            %         for i = indcn
            %             mu(i,j) = 1/(length(indcn));
            %         end
            %     end
            % end
            % for i = 1:R-1
            %     NF(:,:,i) = zeros(n,n);
            %     DF(i,1) = 0;
            %     for j = 1:k
            %         NF(:,:,i) = NF(:,:,i) + mu(i,j)^m*(xn(:,j)-xcn{k}(:,i))*(xn(:,j)-xcn{k}(:,i))';
            %         DF(i,1) = DF(i) + mu(i,j)^m;
            %         F(:,:,i) = NF(:,:,i)/DF(i);
            %     end
            % end
            dist = sqrt((xn(:,k)-xcn{k}(:,R))'*det(F(:,:,R))^(1/n)*inv(F(:,:,R))*...
            (xn(:,k)-xcn{k}(:,R)));
            D(R,k) = pex*(exp(dist)-1) + (1-pex)*dist;
        end
    end
    % Mecanismo de crossover de clusters
    dmin = 1e6;
    Dc = zeros(R,R); % Inicializa as ditâncias
    for i = 1:R
        for j = 1:R % Calcula as distâncias entre centros medidas a partir de cada um
            dist = sqrt((xcn{k}(:,i)-xcn{k}(:,j))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
            (xcn{k}(:,i)-xcn{k}(:,j)));
            Dc(i,j) = pex*(exp(dist)-1) + (1-pex)*dist;
            if (i ~= j) && (Dc(i,j)<dmin) % Encontra a menor das distancias
                i_cp = [i,j];
                dmin = Dc(i,j);
            end
        end
    end
    if dmin < epsilon*r % Se tiver sobreposicao de agrupamentos
        % Fator de ponderacao dos clusters pelo potencial
        rou = Pc{k}(i_cp(1),1)/(Pc{k}(i_cp(1),1)+Pc{k}(i_cp(2),1));
        % Define a nova posicao de centro e elimina os geradores
        xcn{k}(:,R+1) = rou*xcn{k}(:,i_cp(1)) + (1-rou)*xcn{k}(:,i_cp(2));
        xcn{k}(:,max(i_cp)) = [];
        xcn{k}(:,min(i_cp)) = [];
        % Define o novo potencial e elimina os geradores
        Pc{k}(R+1,:) = [rou*Pc{k}(i_cp(1),1)+(1-rou)*Pc{k}(i_cp(2),1),ceil(rou*...
            Pc{k}(i_cp(1),2)+(1-rou)*Pc{k}(i_cp(2),2))];
        Pc{k}(max(i_cp),:) = [];
        Pc{k}(min(i_cp),:) = [];
        % Define a nova taxa de variacao normalizada e elimina os geradores
        tvn_c(R+1) = rou*tvn_c(i_cp(1)) + (1-rou)*tvn_c(i_cp(2));
        tvn_c(max(i_cp)) = [];
        tvn_c(min(i_cp)) = [];
        % Inicializa a linha de distancias do novo cluster e elimina dos
        % geradores
        D(R+1,:) = zeros(size(D(1,:)));
        D(max(i_cp),:) = [];
        D(min(i_cp),:) = [];
        % Inicializa a linha de pertinencias dos novos clusters e elimina
        % dos geradores
        mu(R+1,:) = zeros(size(mu(1,:)));
        mu(max(i_cp),:) = [];
        mu(min(i_cp),:) = [];
        % Determina a nova matriz de covariancia e elimina as geradoras
        NF(:,:,R+1) = rou*NF(:,:,i_cp(1))+(1-rou)*NF(:,:,i_cp(2));
        DF(R+1) = rou*DF(i_cp(1))+(1-rou)*DF(i_cp(2));
        F(:,:,R+1) = NF(:,:,R+1)/DF(R+1);
        NF(:,:,max(i_cp)) = [];
        NF(:,:,min(i_cp)) = [];
        DF(max(i_cp)) = [];
        DF(min(i_cp)) = [];
        F(:,:,max(i_cp)) = [];
        F(:,:,min(i_cp)) = [];
        R = R-1; % Diminui a quantidade de regras
        dist = sqrt((xn(:,k)-xcn{k}(:,R))'*det(F(:,:,R))^(1/n)*inv(F(:,:,R))*...
            (xn(:,k)-xcn{k}(:,R)));
        D(R,k) = pex*(exp(dist)-1) + (1-pex)*dist;
    end
    % Armazena os indices de elementos nulos em D(:,k)
    indcn = find(D(:,k)==0); 
    if isempty(indcn) % Se nao houver elementos nulos
        for i = 1:R
            mu(i,k) = 0;
            for j = 1:R
                mu(i,k) = mu(i,k) + (D(i,k)/D(j,k))^(2/(m-1));
            end
            mu(i,k) = 1/mu(i,k);
        end
    else % Se houver elementos nulos em D(:,k)
        % Divide igualmente a pertinencia nos clusters em que D(i,k)=0,
        % e deixa pertinencia 0 nos demais clusters
        for i = indcn
             mu(i,k) = 1/(length(indcn));
        end
    end
    for i = 1:R % Atualiza a matriz de covariância fuzzy com o dado atual
        NF(:,:,i) = NF(:,:,i) + mu(i,k)^m*(xn(:,k)-xcn{k}(:,i))*(xn(:,k)-xcn{k}(:,i))';
        DF(i,1) = DF(i) + mu(i,k)^m;
        F(:,:,i) = NF(:,:,i)/DF(i);
    end
    qt_reg(k,1) = R;
end
toc
% for l = 1:10
% for j = 1:N
%     for i = 1:R % Medicao da distancia adaptativa do dado atual para os centros
%         dist = sqrt((xn(:,j)-xcn{end}(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
%             (xn(:,j)-xcn{end}(:,i)));
%         D(i,j) = pex*(exp(dist)-1) + (1-pex)*dist;
%     end
%     % Armazena os indices de elementos nulos em D(:,k)
%     indcn = find(D(:,j)==0);
%     if isempty(indcn) % Se nao houver elementos nulos
%         for i = 1:R
%             mu(i,j) = 0;
%             for jj = 1:R
%                 mu(i,j) = mu(i,j) + (D(i,j)/D(jj,j))^(2/(m-1));
%             end
%             mu(i,j) = 1/mu(i,j);
%         end
%     else % Se houver elementos nulos em D(:,k)
%         % Divide igualmente a pertinencia nos clusters em que D(i,k)=0,
%         % e deixa pertinencia 0 nos demais clusters
%         for i = indcn
%             mu(i,j) = 1/(length(indcn));
%         end
%     end
% end
% for i = 1:R
%     NF(:,:,i) = zeros(n,n);
%     DF(i,1) = 0;
%     for j = 1:N
%         NF(:,:,i) = NF(:,:,i) + mu(i,j)^m*(xn(:,j)-xcn{k}(:,i))*(xn(:,j)-xcn{k}(:,i))';
%         DF(i,1) = DF(i) + mu(i,j)^m;
%     end
%     F(:,:,i) = NF(:,:,i)/DF(i);
% end
% end
figure
plot(dados_(1,:),dados_(2,:),'c*',xcn{end}(1,:),xcn{end}(2,:),'r*')
% figure
% plot(dados_(3,:),dados_(4,:),'c*',xcn{end}(3,:),xcn{end}(4,:),'r*')
% figure
% plot(dados_(5,:),dados_(6,:),'c*',xcn{end}(5,:),xcn{end}(6,:),'r*')
% figure
% plot(dados_(7,:),dados_(8,:),'c*',xcn{end}(7,:),xcn{end}(8,:),'r*')
% figure
% plot(dados_(9,:),dados_(10,:),'c*',xcn{end}(9,:),xcn{end}(10,:),'r*')
% % hold on
% % xm = (-0.1:0.005:1.1)';
% % ym = (-0.1:0.005:1.1)';
% for l = 1:R
%     l
%     for i = 1:length(xm)
%         for j = 1:length(ym)
%             dist = sqrt(([xm(i);ym(j)]-xcn{end}(:,l))'*det(F(:,:,l))^(1/n)*inv(F(:,:,l))*...
%             ([xm(i);ym(j)]-xcn{end}(:,l)));
%             Dm(i,j,l) = pex*(exp(dist)-1) + (1-pex)*dist;
%         end
%     end
%     contour(xm,ym,Dm(:,:,l)',0.7*epsilon*r*[1,1],'r');
% end
% Mede a partição final

mu = zeros(R,N);
D = mu;
for j = 1:N
    for i = 1:R % Medicao da distancia adaptativa do dado atual para os centros
        dist = sqrt((xn(:,j)-xcn{end}(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
            (xn(:,j)-xcn{end}(:,i)));
        D(i,j) = pex*(exp(dist)-1) + (1-pex)*dist;
    end
    % Armazena os indices de elementos nulos em D(:,k)
    indcn = find(D(:,j)==0);
    if isempty(indcn) % Se nao houver elementos nulos
        for i = 1:R
            mu(i,j) = 0;
            for jj = 1:R
                mu(i,j) = mu(i,j) + (D(i,j)/D(jj,j))^(2/(m-1));
            end
            mu(i,j) = 1/mu(i,j);
        end
    else % Se houver elementos nulos em D(:,k)
        % Divide igualmente a pertinencia nos clusters em que D(i,k)=0,
        % e deixa pertinencia 0 nos demais clusters
        for i = indcn
            mu(i,j) = 1/(length(indcn));
        end
    end
end
for i = 1:length(x(:,1))
    mx = max(dados(i,:));
    mn = min(dados(i,:));
    if mx == mn
        break
    else
        xc(i,:) = xcn{end}(i,:)*(mx-mn)+mn;
    end
end
dmin = 1e100;
Dc = zeros(R,R); % Inicializa as ditâncias
for i = 1:R
    for j = 1:R % Calcula as distâncias entre centros medidas a partir de cada um
        % Dc(i,j) = norm(xcn{end}(:,i)-xcn{end}(:,j));
        Dc(i,j) = norm(xc(:,i)-xc(:,j));
        if (i ~= j) && (Dc(i,j)<dmin) % Encontra a menor das distancias
            i_cp = [i,j];
            dmin = Dc(i,j);
        end
    end
end
num = 0;
for i = 1:R
    for j = 1:N
        % num = num + mu(i,j)^2*norm(xcn{end}(:,i)-xn(:,j));
        num = num + mu(i,j)^2*norm(xc(:,i)-x(:,j));
    end
end

q_c = num/(N*dmin)

for i = 1:qt_rotulos
    for j = 1:R
        d(j,i) = norm(xcn{end}(:,j)-centros_(:,i));
    end
    d(:,i) = sort(d(:,i));
end
indice_erro_centro = sqrt(mean(d(1,:).^2))