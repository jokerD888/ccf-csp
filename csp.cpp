//202312-1
#include <iostream>
#include <vector>

using namespace std;

int n,m;
vector<vector<int>> arr;

int main()
{
	cin>>n>>m;
	arr.resize(n);
	int t;
	for(int i=0;i<n;++i){
		for(int j=0;j<m;++j){
			cin>>t;
			arr[i].push_back(t);
		}
	}
	vector<int> res(n,0);
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j){
			int k=0;
			for(k=0;k<m;++k)
				if(arr[i][k]>=arr[j][k]) break;
			if(k==m){
				res[i]=j+1;
				break;
			}
			
		}
	}
	for(int i=0;i<n;++i){
		cout<<res[i]<<endl;
	}
	return 0;
}




//202312-2

#include <iostream>
#include <vector>
using namespace std;


int main()
{
	int q;
	cin>>q;
	while(q--){
		long long n;
		int k;
		cin>>n>>k;
		long long res=1;
		
		for(int i=2;i<=n/i;++i){
			if(n%i==0){     
				int s=0;
				while(n%i==0) n/=i,++s;
				if(s>=k)
					while(s--) res*=i;
			}
		}
		cout<<res<<endl;
	}
	return 0;
}

// 202312-3
// 题意：一颗多叉树，对于每颗树，统计它和其全部后代类别的权重之和，同时统计其余全部类别的权重之和，并求二者差值的绝对值，选择绝对值最小的节点
//			然后进行询问，若用户回答是，则保留该节点代表的树，删除其他，否则删除该节点代表的树，保留其他
// 做法模拟：
// 节点：自身权值，自身权值及其后代权值，后代，
// 使用st[x]标记节点x及其后代是否被去除，进行逻辑上的删除
// 流程：读入被测试的类别后，循环{清空待查询集合，dfs更新节点下的权值，查找wsigma最小的节点，从树中去除错误回答的分支}
// 时间大概是2*e6

#include <bits/stdc++.h>
using namespace std;
typedef long long LL;			// ！！！！！！！！！！！！！！！！

const int N=2010;
struct node{
	int w;  // 自身权值
	LL wt;  // 自身及后代权值
	unordered_set<int> sons;		// 后代
};
node tr[N];
bool st[N];     // 此节点及其后代是否被去除
set<int> seg;       // 待查询节点集合

// 更新root节点自身及后代权值
LL dfs(int root){
	seg.insert(root);       // 同时更新待查询集合！！！！！
	LL res=0;
	for(auto c:tr[root].sons){
		if(st[c]) continue;
		res+=dfs(c);        // 递归查询！！！！！！！
	}
	tr[root].wt=res+=tr[root].w;
	return tr[root].wt;
}
// 计算wsigma最小的节点
int query(int root){
	LL wmin=LLONG_MAX,pos=-1;
	for(auto x:seg){
		LL w=abs(tr[root].wt-2*tr[x].wt);		// ！！！！！！！！tr[root].wt-2*tr[x].wt
		if(wmin>w){
			wmin=w;
			pos=x;
		}
	}
	return pos;
}

// 用户询问判断ch是否被归类fa或者fa的后代
bool judge(int fa,int ch){
	if(fa==ch) return true;
	bool flag=false;
	for(auto x:tr[fa].sons){
		flag|=judge(x,ch);      // 递归查询！！！！！！
		if(flag) break;
	}
	return flag;
}

int main()
{
	int n,m,fa;
	cin>>n>>m;
	for(int i=1;i<=n;++i) cin>>tr[i].w;
	for(int i=2;i<=n;++i){
		cin>>fa;
		tr[fa].sons.insert(i);
	}
	for(int i=1;i<=m;++i){
		memset(st,0,sizeof st);		// 每次询问，都先清除删除标记！！！！！！！！！！！！！！！
		int r=1,x;		//r为根节点
		cin>>x;
		while(1){
			seg.clear();		// ！！！！！！！！！！！！！！
			dfs(r);     // 更新权值，同时更新带查询节点集合
			if(seg.size()==1) break;        //直到只剩下一个类别，此时即可确定名词的类别
			int id=query(r);
			cout<<id<<' ';
			if(judge(id,x)) r=id;   // 回答是，保留该类别及其后台		！！！！！！！
			else st[id]=true;       // 否则仅保留其余类别			！！！！！！！
		}
		cout<<endl;
	}
	return 0;
}


//202312-4
// 原文链接：https://blog.csdn.net/cxm_yumu/article/details/137151156
#pragma GCC optimize(2, 3, "Ofast", "inline")
#include <bits/stdc++.h> 
using namespace std;
typedef long long ll;
#define int ll
const int N = 1e5 + 5;
const int M = 350 + 5;
const ll mod = 998244353;//不需要模的时候，把模数调大点即可 
class Mat{
public:
    int v[2][2];
    //vector<vector<int>> v;
    Mat(bool isE = false) {
        v[0][0] = v[1][1] = isE;
        v[1][0] = v[0][1] = 0;
    }
    friend Mat operator * (const Mat& a, const Mat& b) {
        Mat ans;
        ans.v[0][0] = (1ll * a.v[0][0] * b.v[0][0] + 1ll * a.v[0][1] * b.v[1][0]) % mod;
        ans.v[0][1] = (1ll * a.v[0][0] * b.v[0][1] + 1ll * a.v[0][1] * b.v[1][1]) % mod;
        ans.v[1][0] = (1ll * a.v[1][0] * b.v[0][0] + 1ll * a.v[1][1] * b.v[1][0]) % mod;
        ans.v[1][1] = (1ll * a.v[1][0] * b.v[0][1] + 1ll * a.v[1][1] * b.v[1][1]) % mod;
        return ans;
    }
    friend istream& operator >> (istream& in, Mat& x) {
        return in >> x.v[0][0] >> x.v[0][1] >> x.v[1][0] >> x.v[1][1];
    }
    friend ostream& operator << (ostream& out, const Mat& x) {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                out << x.v[i][j] << " ";
        out << endl;
        return out;
    }

};

class OP{
public:
	int type;
	Mat a;
}op[N];

int len;
class Block{
public:
	int li, ri;
	int size; //前缀积个数 
	int del; //记录溢出删除要求数 
	vector<Mat> lsum, rsum; //前缀积 
	void build(int l, int r, bool update){
//		cout << "l = " << l << "r = " << r << endl;
		del = 0, size = 0; 
		deque<int> dq;
		if (update){
			li = l;
			ri = r;	
		}
		for (int i = li; i <= ri; i++){
			if (op[i].type == 3){
				if (dq.size()){
					dq.pop_back();
				} 
				else{
					del++;
				}
			}
			else{
				dq.push_back(i);
			}
		}
		size = dq.size();
		lsum.resize(size + 1), rsum.resize(size + 1);
		lsum[0] = Mat(true);
		rsum[0] = Mat(true);
		int cnt = 0; 
		for (auto i: dq){
			if (op[i].type == 1){
				lsum[cnt + 1] = op[i].a * lsum[cnt];
				rsum[cnt + 1] = rsum[cnt]; //相当于乘以一个单位矩阵 
			}
			else{
				lsum[cnt + 1] = lsum[cnt];
				rsum[cnt + 1] = rsum[cnt] * op[i].a;
			} 
			cnt++;
		}
//		cout << lsum[size] * rsum[size] << endl;
	}
}blk[M];

Mat get_ans(int l, int r){ 
	int lb = l / len;
	int rb = r / len;
	Block ans;
	if (lb == rb){ //说明就是一个块 
		ans.build(l, r, true);
		return ans.lsum[ans.size] * ans.rsum[ans.size]; 
	}
	ans.build(rb * len, r, true);
	swap(ans.lsum[ans.size], ans.lsum[0]);
	swap(ans.rsum[ans.size], ans.rsum[0]);
//	cout << ans.lsum[0] * ans.rsum[0] << endl;
	for (int i = rb - 1; i >= lb + 1; i--){ //注意降序遍历 
        if (blk[i].size <= ans.del) ans.del = ans.del - blk[i].size + blk[i].del;
        else {
            ans.lsum[0] = ans.lsum[0] * blk[i].lsum[blk[i].size - ans.del];
            ans.rsum[0] = blk[i].rsum[blk[i].size - ans.del] * ans.rsum[0];
            ans.del = blk[i].del;
        }
	}
	Block ltmp;
	ltmp.build(l, (lb + 1) * len - 1, true); 
	int need = max(ltmp.size - ans.del, 0ll);
	ans.lsum[0] = ans.lsum[0] * ltmp.lsum[need];
	ans.rsum[0] = ltmp.rsum[need] * ans.rsum[0]; 
//	cout << ans.lsum[0] * ans.rsum[0] << endl;
	return ans.lsum[0] * ans.rsum[0];
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
	int n, m;
	cin >> n >> m;
	len = max(1ll, (int)sqrt(n));
	for (int i = 0; i < n; i++){
		cin >> op[i].type;
		if (op[i].type != 3){
			cin >> op[i].a;
		}
	}
	for (int i = 0; i * len < n; i++){ //初始化各块信息 
		blk[i].build(i * len, min((i + 1) * len - 1, n - 1), true);
	}
	for (int i = 1; i <= m; i++){
		int mode;
		cin >> mode;
		OP new_op;
		if (mode == 1){
			int index;
			cin >> index;
			index--;
			cin >> op[index].type; 
			if (op[index].type != 3){
				cin >> op[index].a;
			}
			int bid = index / len;
			blk[bid].build(0, 0, false);
		}
		else{
			int l, r;
			cin >> l >> r;
			Mat ans = get_ans(--l, --r);
			cout << ans; 
		}
	}
	return 0;
}



// 202309-1
#include <iostream>
using namespace std;

int main()
{
	int n, m;
	cin >> n >> m;
	int x = 0, y = 0;
	int a, b;
	for (int i = 0; i < n; ++i)
	{
		cin >> a >> b;
		x += a, y += b;
	}
	for (int i = 0; i < m; ++i)
	{
		cin >> a >> b;
		cout << a + x << ' ' << b + y << endl;
	}
	return 0;
}

// 202309-2

// 将两种操作分开记录，利用前缀和思想
#include <iomanip>
#include <iostream>
#include <cmath>
using namespace std;
double k[100010];
double ct[100010];

int main()
{
	int n, m;
	cin >> n >> m;
	for (int i = 0; i <= n; ++i)
		k[i] = 1;

	for (int i = 1; i <= n; ++i)
	{
		int t;
		cin >> t;
		if (t == 1)
			cin >> k[i];
		else
			cin >> ct[i];
		k[i] *= k[i - 1];
		ct[i] += ct[i - 1];
	}
	for (int i = 1; i <= m; ++i)
	{
		int l, r;
		double x, y, a, b;
		cin >> l >> r >> x >> y;

		x *= (k[r] / k[l - 1]);
		y *= (k[r] / k[l - 1]);
		double c = ct[r] - ct[l - 1];
		a = x * cos(c) - y * sin(c);
		b = y * cos(c) + x * sin(c);

		printf("%.5f %.5f\n", a, b);
	}
	return 0;
}



// 	202309-3 梯度求解
// 题意：给一个逆波兰序表达式，对该式子求偏导，计算结果
// 思路：逆波兰序，二叉树的后缀序列，树就是一个有递归结构的表示形式，要求整个树的偏导值，
//      先看根节点操作符，假设是乘法，那我们需要求左子树的函数值和偏导值，右子树的函数值和偏导值，
//		然后根据上述求导的式子计算得到当前子树的函数值和偏导值，而求左右子树的函数值和偏导值实际上就是一个子问题，因此就可以递归下去了。

#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
#define CONST -1
#define VAR -2
#define OP -3
const int mo=1e9+7;

int n,m;
vector<int> l,r,info,kind;
stack<int> id;
int main(){

	string s;
	cin>>n>>m;
	getchar();      // '\n'
	getline(cin,s);
	stringstream ss(s);
	
	// 建立表达式树
	int node_id=0;
	while(getline(ss,s,' ')){		// OP
		if(s.size()==1 && (s[0]=='+' || s[0]=='-' || s[0]=='*')){
			int rson=id.top();		// ！！！！！！！！！
			id.pop();
			int lson=id.top();		// ！！！！！！！！！
			id.pop();
			l.push_back(lson);
			r.push_back(rson);
			info.push_back(s[0]);
			kind.push_back(OP);
			id.push(node_id++);
		}else if(s[0]=='x'){		// VAR
			int x=stoi(s.substr(1));
			--x;        // 数组a下标从0开始
			l.push_back(-1);
			r.push_back(-1);
			info.push_back(x);
			kind.push_back(VAR);
			id.push(node_id++);
		}else{					// CONST
			int x=stoi(s);
			l.push_back(-1);
			r.push_back(-1);
			info.push_back(x);
			kind.push_back(CONST);
			id.push(node_id++);
		}
	}
	int root=id.top();		// ！！！！！！！！！！！！
	vector<int> a(n);	// ！！！！！！！！！！！！
	
	// 计算u子树的值及其偏导数值
	 function<array<int, 2>(LL, LL)> solve = [&](int u, int x){
       if (kind[u] == VAR){
           return array<int, 2>{a[info[u]], (info[u] == x)};
       }else if (kind[u] == CONST){
           return array<int, 2>{info[u], 0};
       }else{
           auto lans = solve(l[u], x), rans = solve(r[u], x);
           int sum = 0, dsum = 0;
           if (info[u] == '+'){
               sum = lans[0] + rans[0];
               dsum = lans[1] + rans[1];
           }else if (info[u] == '-'){
               sum = lans[0] - rans[0];
               dsum = lans[1] - rans[1];
           }else{
               sum = 1ll * lans[0] * rans[0] % mo;
               dsum = (1ll * lans[0] * rans[1] % mo + 1ll * lans[1] * rans[0] % mo);
           }
			sum=((LL)sum+mo)%mo;
			dsum=((LL)dsum+mo)%mo;
		
           return array<int, 2>{sum, dsum};
       }
   };
	
	for(int i=1;i<=m;++i){
		int x;
		cin>>x;		// 自变量
		--x;
		for(auto &i:a)
			cin>>i;
		cout<<solve(root,x)[1]<<endl;
	}
	return 0;
}


// 202309-4 阴阳龙
// 题意：在n*m的矩形中，有p个不同的点，龙出现q次，每次给出龙出现的坐标和强度，龙从八个方向向外走，若先遇到边界，此次出现无影响，若先遇到人，则按照任何龙之间的距离k，将八个方向距离为k的人，按照龙出现时的强度进行旋转
// 思路：使用unordered_map<int, set<array<int, 2>>>，分别映射，相同的横，纵，斜率为-1，斜率为1的点的集合，其中利用set有序的特性，二分找到其所在方向最近的人，各个方向找完后，放入数组，按照距离进行排序，选最小的距离进行旋转
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
// 上 右上 -> 顺时针
const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};      // 题目给的
const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

int main(){
    ios::sync_with_stdio(false);        // 输入较多，优化一下
    cin.tie(0);
    cout.tie(0);
    int n, m, p, q;
    cin >> n >> m >> p >> q;
    
    vector<array<int, 2>> pos(p);
    // 四个哈希表，分别映射，相同的横，纵，斜率为-1，斜率为1的点的集合，其中利用set有序的特性
    unordered_map<int, set<array<int, 2>>> row, col, ld, rd;        // ！！！！！！！！！！！！！！！！！！！！
    
    auto insert = [&](int id){
        int x = pos[id][0], y = pos[id][1];
        row[x].insert({y, id});
        col[y].insert({x, id});
        ld[x + y].insert({y, id});
        rd[x - y].insert({y, id});
    };
    auto remove = [&](int id){
        int x = pos[id][0], y = pos[id][1];
        row[x].erase({y, id});
        col[y].erase({x, id});
        ld[x + y].erase({y, id});
        rd[x - y].erase({y, id});
    };
    for(int i = 0; i < p; ++ i){
        cin >> pos[i][0] >> pos[i][1];
        insert(i);
    }
    for(int i = 0; i < q; ++ i){
        int u, v, t;
        cin >> u >> v >> t;
        vector<array<int, 3>> candidate;// 距离，id,方向

        auto search = [&](const set<array<int, 2>>& people, int d, int dirr, int dirl){
            auto pos = people.lower_bound(array<int, 2>{d, p}); // [2]设为p,使得lower_bound找到大于d的第一个！！！！
            if (pos != people.end()){       // ！！！！！！！！
                candidate.push_back({(*pos)[0] - d, (*pos)[1], dirr});
            }
            if (pos != people.begin()){
                pos = prev(pos);    // 往前一下
                if ((*pos)[0] == d && pos != people.begin())    // 不能重合，能向前了，再往前一下
                    pos = prev(pos);

                if ((*pos)[0] != d){
                    candidate.push_back({d - (*pos)[0], (*pos)[1], dirl});
                }
            }
        };

        search(row[u], v, 2, 6);    // 2 6 分别代表了x轴的正方向和x轴的负方向,一下类似
        search(col[v], u, 0, 4);
        search(ld[u + v], v, 3, 7);
        search(rd[u - v], v, 1, 5);

        if (candidate.empty())  // 没人，跳
            continue;
        sort(candidate.begin(), candidate.end(), [&](const array<int, 3>& a, const array<int, 3>& b){
            return a[0] < b[0];
        });
        int mindis = min({u - 1, n - u, v - 1, m - v}); // 到达边界最小距离
        if (candidate[0][0] > mindis)   // 人太远，跳
            continue;
        mindis = candidate[0][0];
        for(int i = 0; i < candidate.size(); ++ i){
            if (candidate[i][0] != mindis)
                break;
            // 旋转
            int dis = candidate[i][0];
            int id = candidate[i][1];
            remove(id);     // ！！！！ 删除旧位置
            int dir = (candidate[i][2] + t) % 8;        // ！！！！！！！！！！ 按题目
            pos[id][0] = u + dis * dx[dir];
            pos[id][1] = v + dis * dy[dir];
            insert(id);     // !!!!! 插入新位置
        }
    }
    LL ans = 0;
    for(int i = 0; i < p; ++ i){
        ans ^= (1ll * (i + 1) * pos[i][0] + pos[i][1]);     // ^ 异或
    }
    cout << ans << '\n';
    return 0;
}



// 202305-1

#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

int main()
{
	int n;
	cin >> n;

	unordered_map<string, int> hs;
	while (n--)
	{
		string s, tmp;
		for (int i = 0; i < 8; ++i)
		{
			cin >> tmp;
			s += tmp;
		}
		cout << ++hs[s] << endl;
	}
	return 0;
}

// 202305-2

#include <iostream>
#include <vector>
using namespace std;

typedef long long LL;
const int N = 10010, D = 21;
int n, d;
int Q[N][D], K[N][D], V[N][D], W[N];
LL tmp1[D][D], tmp2[N][D];

//
// n*d * d*n * n*d

int main()
{
	cin >> n >> d;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < d; ++j)
			cin >> Q[i][j];
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < d; ++j)
			cin >> K[i][j];
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < d; ++j)
			cin >> V[i][j];
	for (int i = 0; i < n; ++i)
		cin >> W[i];

	// K(T)*V = tmp1
	for (int i = 0; i < d; ++i)
		for (int j = 0; j < d; ++j)
			for (int p = 0; p < n; ++p)
				tmp1[i][j] += K[p][i] * V[p][j];
	// Q*tmp1=tmp2
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < d; ++j)
		{
			for (int p = 0; p < d; ++p)
				tmp2[i][j] += Q[i][p] * tmp1[p][j];
			cout << tmp2[i][j] * W[i];
			if (j != d - 1)
				cout << ' ';
		}
		cout << endl;
	}

	return 0;
}



// 202305-3 解压缩
#include <iostream>
#include <cmath>
#include <string>
using namespace std;

int n;
int p = 0;     // p为str的当前索引
string s, str, ans;
int read_bit() {		// 读一字节
    int res = 16 * (isdigit(str[p]) ? str[p] - '0' : str[p] - 'a' + 10);        // !!!!!!!!! +10别漏了
    res += (isdigit(str[p + 1]) ? str[p + 1] - '0' : str[p + 1] - 'a' + 10);
    p += 2;
    return res;
}
void read_back(int o, int l) {
    string ss;
    l *= 2;
    while (l) {		// ！！！！！！！！
        for (int i = ans.size() - o * 2; l && i < ans.size(); ++i, --l)
            ss += ans[i];
    }
    ans += ss;

}
void prase() {
    while (p < n * 2) {		// 一字节两个16位进制
        int flag = read_bit();
        if ((flag & 3) == 0) {        // 字面量，最低两位为0
            int l = (flag >> 2) + 1;        // (l-1)
            if (l > 60) {
                int t = l - 60;
                l = 1;      // ！！！！！！！！！！！！！！！！！！   (l-1)
                for (int i = 0; i < t; ++i)
                    l += read_bit() * pow(256, i);
            }
            for (int i = 0; i < l * 2; ++i, ++p)
                ans += str[p];
        } else if ((flag & 3) == 1) {     // ！！！！！！！！！！！ 注意位运算的优先级
            int num1 = flag, num2 = read_bit();//o其低 8 位存储于随后的字节中，高 3 位存储于首字节的高 3 位中。l存储于首字节的 2 至 4 位
            int o = num2 + (num1 >> 5)*256, l = 4 + ((num1 >> 2) & 7);      // *256（高位）     4+
            read_back(o, l);
        } else if ((flag & 3) == 2) {
       		// o存储在随后的两个字节，注意是小端，l-1高六位	
            int l = (flag >> 2) + 1, o = read_bit() + read_bit() * 256;    // !!!!!!!!!!!!!  (l-1)
            read_back(o, l);
        }
    }
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin >> n;
    for (int i = 0; i < (n+7) / 8; ++i) cin >> s, str += s;     // !!!!!!!!!!!!  (n+7)上取整

    // 读走长度信息，实际的长度不用计算出来，没用
    // 长度字段，除了最后一个字节的最高位为 0，其余字节的最高位为 1
    for (int i = 0, bt=128; bt >= 128; ++i)     // bt128!!!!!!!!!!
        bt = read_bit();
    prase();

    for (int i = 0; i < ans.size(); ++i) {
        if (i && i % 16 == 0) cout << '\n';
        cout << ans[i];
    }

    return 0;
}


//202305-4
// 原文链接：https://blog.csdn.net/qq_45123552/article/details/136783152
#pragma GCC optimize(2, 3, "Ofast", "inline")
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'

using i64 = long long;
using ui64 = unsigned long long;
using i128 = __int128;
#define inf (int)0x3f3f3f3f3f3f3f3f
#define INF 0x3f3f3f3f3f3f3f3f
#define yn(x) cout << (x ? "yes" : "no") << endl
#define Yn(x) cout << (x ? "Yes" : "No") << endl
#define YN(x) cout << (x ? "YES" : "NO") << endl
#define mem(x, i) memset(x, i, sizeof(x))
#define cinarr(a, n) for (int i = 1; i <= n; i++) cin >> a[i]
#define cinstl(a) for (auto& x : a) cin >> x;
#define coutarr(a, n) for (int i = 1; i <= n; i++) cout << a[i] << " \n"[i == n]
#define coutstl(a) for (const auto& x : a) cout << x << ' '; cout << endl
#define all(x) (x).begin(), (x).end()
#define md(x) (((x) % mod + mod) % mod)
#define ls (s << 1)
#define rs (s << 1 | 1)
#define ft first
#define se second
#define pii pair<int, int>
#ifdef DEBUG
    //#include "debug.h"
#else
    #define dbg(...) void(0)
#endif

const int N = 1e5 + 5;
const int mod = 998244353;

int w[N][10];
int n, m, t, k, q;
map<int, int> mp[N];
struct edge { int u, v; vector<vector<int>> w; } es[N];
int del[N], in[N];

void work() {
    cin >> n >> m >> k;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            cin >> w[i][j];
    for (int i = 0; i < m; i++) {
        auto& e = es[i];
        cin >> e.u >> e.v;
        e.w.assign(k, vector<int>(k));
        for (auto& v : e.w)
            for (auto& x : v) cin >> x;
        mp[e.u][e.v] = i;
        mp[e.v][e.u] = i;
        ++in[e.u], ++in[e.v];
    }

    queue<int> q;
    for (int i = 0; i < n; i++) if (in[i] <= 2) q.push(i);
    while (!q.empty()) {
        auto transposi = [&] (auto& e) {
            swap(e.u, e.v);
            for (int i = 0; i < k; i++)
                for (int j = i + 1; j < k; j++)
                    swap(e.w[i][j], e.w[j][i]);
        };
        auto del1 = [&] (int u) {
            assert(mp[u].size() == 1);
            int v = mp[u].begin()->first;
            auto& e = es[mp[u].begin()->second];
            if (u == e.v) transposi(e);
            for (int j = 0; j < k; j++) {
                int mn = inf;
                for (int i = 0; i < k; i++)
                    mn = min(mn, e.w[i][j] + w[u][i]);
                w[v][j] += mn;
            }
            del[u] = 1;
            mp[u].clear();
            mp[v].erase(u);
            if (--in[v] <= 2) q.push(v);
        };
        auto del2 = [&] (int u) {
            assert(mp[u].size() == 2);
            int v1 = mp[u].begin()->first, v2 = mp[u].rbegin()->first;
            auto& e = mp[v1].count(v2) ? es[mp[v1][v2]] : (++in[v1], ++in[v2], es[mp[v1][v2] = mp[v2][v1] = m++] = {v1, v2, vector<vector<int>>(k, vector<int>(k, 0))});
            if (v1 == e.v) swap(v1, v2);
            auto& e1 = es[mp[u][v1]], & e2 = es[mp[u][v2]];
            if (u == e1.v) transposi(e1);
            if (u == e2.v) transposi(e2);
            for (int j = 0; j < k; j++)
                for (int l = 0; l < k; l++) {
                    int mn = inf;
                    for (int i = 0; i < k; i++)
                        mn = min(mn, e1.w[i][j] + e2.w[i][l] + w[u][i]);
                    e.w[j][l] += mn;
                }
            del[u] = 1;
            mp[u].clear();
            mp[v1].erase(u), mp[v2].erase(u);
            if (--in[v1] <= 2) q.push(v1);
            if (--in[v2] <= 2) q.push(v2);
        };

        int u = q.front(); q.pop();
        if (del[u]) continue;
        if (in[u] == 0) break;
        else if (in[u] == 1) del1(u);
        else del2(u);
    }

    vector<int> vv;
    vector<edge> ee;
    for (int i = 0; i < n; i++) if (!del[i]) vv.emplace_back(i);
    for (int i = 0; i < m; i++)
        if (!del[es[i].u] && !del[es[i].v])
            ee.push_back({lower_bound(all(vv), es[i].u) - vv.begin(), lower_bound(all(vv), es[i].v) - vv.begin(), es[i].w});
    int nn = vv.size(), mm = ee.size();
    int ans = inf;
    vector<int> sel(nn);
    function<void(int, int)> dfs = [&] (int i, int sum) {
        if (i == nn) {
            int tem = sum;
            for (int j = 0; j < mm; j++)
                tem += ee[j].w[sel[ee[j].u]][sel[ee[j].v]];
            ans = min(ans, tem);
            return;
        }
        for (int j = 0; j < k; j++) {
            sel[i] = j;
            dfs(i + 1, sum + w[vv[i]][j]);
        }
    };
    dfs(0, 0);
    cout << ans << endl;
}

int main() {

    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int Case = 1;
    while (Case--) work();
    return 0;
}




// 202303-1 
//由于矩阵乘法的结合律，可以从右边向左算
#include <bits/stdc++.h>
	using namespace std;

int main()
{
	int n, a, b;
	cin >> n >> a >> b;
	int res = 0;
	for (int i = 1; i <= n; ++i)
	{
		int x1, y1, x2, y2;
		cin >> x1 >> y1 >> x2 >> y2;
		x1 = max(0, x1);
		x2 = max(0, x2);
		y1 = max(0, y1);
		y2 = max(0, y2);
		x1 = min(x1, a), x2 = min(x2, a);
		y1 = min(y1, b), y2 = min(y2, b);
		res += (x2 - x1) * (y2 - y1);
	}
	cout << res;
	return 0;
}

//202303-2
#include <bits/stdc++.h>
	using namespace std;
const int N = 1e5 + 10;
int cnt[N], sum[N];
int a[N];
int main()
{
	int n, m, k;
	cin >> n >> m >> k;
	for (int i = 1; i <= n; ++i)
	{
		int t;
		cin >> a[i] >> t;
		cnt[a[i]] += t;
	}
	sort(a + 1, a + 1 + n);
	n = unique(a + 1, a + 1 + n) - a - 1;
	for (int i = n; i >= 0; --i)
	{
		sum[i] = sum[i + 1] + cnt[a[i]];
	}
	int ans = 0;

	for (int i = n; i > 0; --i)
	{
		int t = a[i] - a[i - 1];
		if (m / sum[i] >= t)
		{
			m -= sum[i] * t;
		}
		else
		{
			ans = a[i] - m / sum[i];
			break;
		}
	}
	cout << max(k, ans);
	return 0;
}

// 202303-3 LDAP
#include <bits/stdc++.h>
using namespace std;

const int N = 2510;

int q[N];//id(DN)
map<pair<int, int>, vector<int>>has;    //map[{i,j}]=p  属性为i，属性值为j的用户的集合
unordered_map<int, vector<int>>hass;    // map[i]为属性编号为i的用户的集合
int n;

bitset<2501> fun(string s) {
    bitset<2501>ans;
    // &/| (exp1)(exp2)
    if (s[0] == '&' || s[0] == '|') {
        int l1 = 1;
        int l2 = 0;
        int count = 1;
        // 找到第一个表达式的右括号
        for (int i = l1 + 1; i < s.size(); ++i) {       
            if (s[i] == '(') count++;
            else if (s[i] == ')') count--;
            if (count == 0) {
                l2 = i;
                break;
            }
        }
        int r1 = l2 + 1;
        int r2 = s.size() - 1;
        bitset<2501> l = fun(s.substr(l1 + 1, l2 - l1 - 1));
        bitset<2501> r = fun(s.substr(r1 + 1, r2 - r1 - 1));

        if (s[0] == '&') {
            ans = l & r;
        } else {
            ans = l | r;
        }
        return ans;
    } else {        //属性编号:/~属性值
        int i = s.find(':');
        bool flag = 0;
        if (i == -1) {
            i = s.find('~');
            flag = 1;
        }
        int l = stoi(s.substr(0, i));
        int r = stoi(s.substr(i + 1, s.size() - i - 1));

        // 先不管是断言和反断言，先直接取1
        for (auto x : has[{l, r}]) {
            ans[x] = 1;
        }
        if (flag)   // 若是反断言，全部取反
            for (auto x : hass[l]) {
                ans.flip(x);		//!!!!!!!!!!!!!!!!!!
            }

    }
    return ans;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);

    cin >> n;
    for (int i = 1; i <= n; ++i) {
        cin >> q[i];
        int k; cin >> k;
        while (k--) {
            int x, y; cin >> x >> y;
            has[{x, y}].push_back(i);
            hass[x].push_back(i);
        }
    }

    int m; cin >> m;
    while (m--) {
        string s; cin >> s;
        bitset<2501>ans = fun(s);           // 每一位是一个用户真或较假
        vector<int>v;
        for (int i = 1; i <= n; ++i) {
            if (ans[i]) v.push_back(q[i]);
        }
        sort(v.begin(), v.end());
        for (auto x : v)
            cout << x << " ";
        cout << '\n';
    }
    return 0;
}

//202303-4	星际网络II
// 原文链接：https://blog.csdn.net/AC__dream/article/details/130851714
#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<map>
#include<queue>
#include<vector>
#include<cmath>
using namespace std;
const int N=2e6+10;
int l[N],r[N],mx[N],mn[N],lz[N],s[N];
void pushup(int id)
{
	mx[id]=max(mx[id<<1],mx[id<<1|1]);
	mn[id]=min(mn[id<<1],mn[id<<1|1]);
	s[id]=s[id<<1]+s[id<<1|1];
}
void pushdown(int id)
{
	if(lz[id]!=0x3f3f3f3f)
	{
		s[id<<1]=r[id<<1]-l[id<<1]+1;
		s[id<<1|1]=r[id<<1|1]-l[id<<1|1]+1;
		mx[id<<1]=lz[id];
		mx[id<<1|1]=lz[id];
		mn[id<<1]=lz[id];
		mn[id<<1|1]=lz[id];
		lz[id<<1]=lz[id];
		lz[id<<1|1]=lz[id];
		lz[id]=0x3f3f3f3f;
	}
}
void build(int id,int L,int R)
{
	l[id]=L;r[id]=R;mn[id]=0x3f3f3f3f;mx[id]=-0x3f3f3f3f;lz[id]=0x3f3f3f3f;
	if(L>=R) return ;
	int mid=L+R>>1;
	build(id<<1,L,mid);
	build(id<<1|1,mid+1,R);
	pushup(id);
}
void update_interval(int id,int L,int R,int val)
{
	if(l[id]>=L&&r[id]<=R)
	{
		s[id]=r[id]-l[id]+1;
		mx[id]=val;
		mn[id]=val;
		lz[id]=val;
		return ;
	}
	pushdown(id);
	int mid=l[id]+r[id]>>1;
	if(mid>=L) update_interval(id<<1,L,R,val);
	if(mid+1<=R) update_interval(id<<1|1,L,R,val);
	pushup(id);
}
int query_mx(int id,int L,int R)
{
	if(l[id]>=L&&r[id]<=R) return mx[id];
	pushdown(id);
	int mid=l[id]+r[id]>>1;
	int ans=-0x3f3f3f3f;
	if(mid>=L) ans=query_mx(id<<1,L,R);
	if(mid+1<=R) ans=max(ans,query_mx(id<<1|1,L,R));
	return ans;
}
int query_mn(int id,int L,int R)
{
	if(l[id]>=L&&r[id]<=R) return mn[id];
	pushdown(id);
	int mid=l[id]+r[id]>>1;
	int ans=0x3f3f3f3f;
	if(mid>=L) ans=query_mn(id<<1,L,R);
	if(mid+1<=R) ans=min(ans,query_mn(id<<1|1,L,R));
	return ans;
}
int query_sum(int id,int L,int R)
{
	if(l[id]>=L&&r[id]<=R) return s[id];
	pushdown(id);
	int mid=l[id]+r[id]>>1;
	long long ans=0;
	if(mid>=L) ans+=query_sum(id<<1,L,R);
	if(mid+1<=R) ans+=query_sum(id<<1|1,L,R);
	return ans;
}
int n,q;
vector<string> alls;
string add(string s)
{
	int flag=1;
	string t=s;
	for(int i=s.size()-1;i>=0;i--)
	{
		if(s[i]==':') continue;
		else if(s[i]=='f'&&flag)
			t[i]='0';
		else
		{
			if(s[i]=='9') t[i]='a';
			else t[i]=s[i]+1;
			break;
		}
	}
	return t;
}
int find(string s)
{
	return lower_bound(alls.begin(),alls.end(),s)-alls.begin()+1;
}
struct node{
	int op;
	int id;
	string l,r;
}p[N];
int main()
{
	cin>>n>>q;
	for(int i=1;i<=q;i++)
	{
		scanf("%d",&p[i].op);
		if(p[i].op==1)
		{
			cin>>p[i].id>>p[i].l>>p[i].r;
			alls.push_back(p[i].l);
			alls.push_back(p[i].r);
			alls.push_back(add(p[i].r));
		}
		else if(p[i].op==2)
		{
			cin>>p[i].l;
			alls.push_back(p[i].l);
		}
		else
		{
			cin>>p[i].l>>p[i].r;
			alls.push_back(p[i].l);
			alls.push_back(p[i].r);
			alls.push_back(add(p[i].r));
		}
	}
	sort(alls.begin(),alls.end());
	alls.erase(unique(alls.begin(),alls.end()),alls.end());
	build(1,1,alls.size());
	for(int i=1;i<=q;i++)
	{
		if(p[i].op==1)
		{
			int ll=find(p[i].l),rr=find(p[i].r);
			if(query_mn(1,ll,rr)==0x3f3f3f3f)//该块土地全部未被分配 
			{
				puts("YES");
				update_interval(1,ll,rr,p[i].id);
			}
			else if(query_mn(1,ll,rr)==p[i].id&&query_mx(1,ll,rr)==p[i].id)//该块土地只分配给了一个人 
			{
				if(query_sum(1,ll,rr)==(rr-ll+1))//该块土地本来就已经全部分配给了p[i].id 
					puts("NO");
				else
				{
					puts("YES");
					update_interval(1,ll,rr,p[i].id);
				} 
			}
			else//该块土地已经分配给了除了p[i].id以外的人，所以无法再分配给p[i].id 
				puts("NO");
		}
		else if(p[i].op==2)
		{
			int ll=find(p[i].l);
			int t=query_mx(1,ll,ll);
			if(t!=-0x3f3f3f3f)
				printf("%d\n",t);
			else
				printf("0\n");
		}
		else
		{
			int ll=find(p[i].l),rr=find(p[i].r);
			int id=query_mn(1,ll,rr);
			if(id==query_mx(1,ll,rr)&&query_sum(1,ll,rr)==(rr-ll+1))//该块土地只分配给了一个人 
				printf("%d\n",id);
			else
				printf("0\n");
		}
	}
	return 0;
}


// 202212-1
#include <iostream>
#include <cmath>
using namespace std;

int n;
double f;
int main()
{
	cin>>n>>f;
	double res=0;
	cin>>res;
	for(int i=1;i<=n;++i){
		double x;
		cin>>x;
		res+=x*pow(1+f,-i);
	}
	cout<<res<<endl;
	return 0;
}




// 202212-2
#include <bits/stdc++.h>
using namespace std;
const int N=110;
int n,m;

vector<int> ne[N];
int in[N],res[N],res1[N],pre[N],w[N];
int main()
{
	cin>>n>>m;
	for(int i=1;i<=m;++i) {
		// 由于0不用，故可避免if处理
		cin>>pre[i];
		ne[pre[i]].push_back(i);
		//++in[pre[i]];

	}
	for(int i=1;i<=m;++i) cin>>w[i];
	
	bool ok=true;
	// 最早
	for(int i=1;i<=m;++i){
		if(pre[i]){         // 有前驱
			res[i]=res[pre[i]]+w[pre[i]];
		}
       cout<<res[i]+1<<' ';    // 从第一天开始，而res是第0天开始，故+1
		if(res[i]+w[i]>n) ok=false;
	}
	cout<<endl;
	
	// 最晚
	if(ok){
		for(int i=m;i>0;--i){
			res1[i]=n-w[i];
			if(!ne[i].empty()){  // 有后继
				for(int j=0;j<ne[i].size();++j) res1[i]=min(res1[ne[i][j]]-w[i],res1[i]);
			}
		}
		for(int i=1;i<=m;++i)
		cout<<res1[i]+1<<' ';
	}
	
	return 0;
}



// 202212-3


#include <bits/stdc++.h>
using namespace std;


int Q[8][8], M[8][8], MQ[8][8], R[8][8], arr[64];
int n;
const double PI = acos(-1);
// 不如直接看图定位
void readM() {
    int r = 0, c = 0, i = 0;
    M[r][c] = arr[i++]; M[r][++c] = arr[i++];
    while (i < 64) {
        while (c - 1 >= 0 && r + 1 <= 7) M[++r][--c] = arr[i++];
        if (r == 7) M[r][++c] = arr[i++];
        else M[++r][c] = arr[i++];
        if (i == 64) break;

        while (r - 1 >= 0 && c + 1 <= 7) M[--r][++c] = arr[i++];
        if (r==0)  M[r][++c] = arr[i++];
        else M[++r][c] = arr[i++];
    }
}
void Print(int Mtrix[][8]) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j)
            cout << Mtrix[i][j] << ' ';
        cout << '\n';
    }
}
void MmutQ() {
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            MQ[i][j] = M[i][j] * Q[i][j];
}
double Alpa(int u) {
    return u ? 1 : pow(0.5, 0.5);
}
void Prase() {
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            double t = 0;
            for (int x = 0; x < 8; ++x)
                for (int y = 0; y < 8; ++y) {
                    t += Alpa(x) * Alpa(y) * MQ[x][y] * cos(PI / 8 * (i + 0.5) * x) * cos(PI / 8 * (j + 0.5) * y);
                }
            R[i][j] = t / 4+128+0.5;        // !!!!!!!!!!!!!!!!!!!!! 四舍五入
            R[i][j] = max(0, R[i][j]);
            R[i][j] = min(255, R[i][j]);
        }
}

int main() {
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            cin >> Q[i][j];
    int T;
    cin >> n >> T;
    for (int i = 0; i < n; ++i) cin >> arr[i];

    readM();
    MmutQ();
    Prase();

    if (T == 0) Print(M);
    else if (T == 1) Print(MQ);
    else Print(R);

    return 0;
}


//202212-4

// 思路：启发式合并，使用multiset<int> 存储休假时间，对于每个节点将轻儿子合并到重儿子上
// 对于每个节点，先计算出其所有的轻儿子，再计算重儿子，然后所有轻儿子的下属集合加到重儿子集合中

#pragma GCC optimize(3)

#include <algorithm>
#include <array>
#include <iostream>
#include <set>

using namespace std;

const int N = 300010;
const int inf = 1e9+10;

struct node {
    int from, to;
} E[N * 2];
int a[N], n, idx, h[N];
int siz[N], son[N];     // son[i]记录i节点最重的儿子
long long sum, ans[N];
multiset<int> S;

void add_edge(int u, int v) {
    E[++idx] = { v, h[u] }, h[u] = idx;
}

void set_init() {
    S.insert(inf), S.insert(inf);
    S.insert(-inf), S.insert(-inf);
}

// 预处理出每个节点的下属个数，深度，最终的儿子
void dfs_init(int u, int par = 0) {
    siz[u] = 1;

    for (int i = h[u]; i; i = E[i].to) {
        int v = E[i].from;
        if (v == par) {
            continue;
        }
        dfs_init(v, u);
        siz[u] += siz[v];
        if (siz[v] > siz[son[u]]) {
            son[u] = v;
        }
    }
}

array<int, 5> b{};
// x不是哨兵
bool check(int x) {
    return x != inf && x != -inf;
}

long long calc(long long A, long long B, long long C) {
    if (!(check(B))) {      // 中间不是有意义的数据，return        // !!!!!!!!!!!!!
        return 0;
    } else if (check(A) && check(C)) {
        return min(1LL * (B - A) * (B - A), 1LL * (C - B) * (C - B));
    } else if (!check(A) && check(C)) {
        return 1LL * (C - B) * (C - B);
    } else if (check(A) && !check(C)) {
        return 1LL * (B - A) * (B - A);
    } else {        // !!!!!!!!!!!!!
        return 0;
    }
}

void add(int u) {
    auto it = S.lower_bound(a[u]);
    it--;
    it--;

    b[0] = *(it++);
    b[1] = *(it++);
    b[2] = a[u];
    b[3] = *(it++);
    b[4] = *it;
    // 先减去原先的贡献
    sum -= calc(b[0], b[1], b[3]);
    sum -= calc(b[1], b[3], b[4]);
    // 再计算新的贡献
    for (int i = 1; i < 4; i++) {
        sum += calc(b[i - 1], b[i], b[i + 1]);
    }
    S.insert(a[u]);
}

void add_to_subtree(int u, int par) {       // 将u的所有下属加到树上
    add(u);
    for (int i = h[u]; i; i = E[i].to) {
        int v = E[i].from;
        if (v == par) {
            continue;
        }
        add_to_subtree(v, u);
    }
}

void dfs(int u, int par = 0, bool keep = 0) {
    for (int i = h[u]; i; i = E[i].to) {
        int v = E[i].from;
        if (v == par || v == son[u]) {  // 先计算所有的轻儿子
            continue;
        }
        dfs(v, u, 0);
    }

    if (son[u]) {       // 再计算重儿子，重儿子的集合保留，不清楚
        dfs(son[u], u, 1);
    }

    add(u);             // 加上自己！！！！！！！！！！！！！！！！！
    for (int i = h[u]; i; i = E[i].to) {        // 加上所有轻儿子的集合
        int v = E[i].from;
        if (v != son[u] && v != par) {
            add_to_subtree(v, u);
        }
    }

    ans[u] = sum;           // ！！！！！！！！！！！！！！保存结果
    if (!keep) {            // 如果不是重儿子，则不用保留，清空集合
        S.clear();
        set_init();
        sum = 0;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin >> n;
    for (int i = 2; i <= n; i++) {
        int p;
        cin >> p;
        add_edge(p, i);
        add_edge(i, p);
    }

    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }

    dfs_init(1);
    set_init();
    dfs(1);

    for (int i = 1; i <= n; i++) {
        cout << ans[i] << "\n";
    }
}



// 202209-1
#include <iostream>
using namespace std;

int n,m;
int a[25];  // 前缀乘

int main()
{
	cin>>n>>m;
	a[0]=1;
	int last=0;
	for(int i=1;i<=n;++i) {
		cin>>a[i];
		a[i]*=a[i-1];
		cout<<(m%a[i]-m%a[i-1])/a[i-1]<<' ';

	}
	

	return 0;
}


//202209-2
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int n,x;
int a[32];

int ans=0x3f3f3f3f;
void dfs(int i,int res){
	if(res<x) return;
	if(i>n+1) return;
	ans=min(ans,res);
	dfs(i+1,res);
	dfs(i+1,res-a[i]);
}

int main()
{
	cin>>n>>x;

	for(int i=1;i<=n;++i){
		cin>>a[i];
		a[0]+=a[i];
	}
	dfs(1,a[0]);
	cout<<ans;
	return 0;
}

// dp版本
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int n,x;
int a[32];

int dp[31][300005];
int main()
{
	cin>>n>>x;

	for(int i=1;i<=n;++i){
		cin>>a[i];
		a[0]+=a[i];
	}
	dp[0][a[0]]=1;
	for(int i=1;i<=n;++i){
		for(int j=x;j<=a[0];++j){
			dp[i][j-a[i]]=dp[i][j-a[i]] || dp[i-1][j] && j-a[i]>=x;  // 不要
			dp[i][j]=dp[i-1][j];        // 要
		}
	}
	for(int i=x;i<=a[0];++i)
		if(dp[n][i]){
			 cout<<i;
			 break;
		}

	return 0;
}

// dp优化,滚动数组
#include <bits/stdc++.h>
using namespace std;

const int N=35,M=300010;
int a[N];
bool f[M];
int main()
{
   int n,m;
   cin>>n>>m;
   f[0]=true;
   for(int i=1;i<=n;++i) cin>>a[i];
   for(int i=1;i<=n;++i){
       for(int j=M;j>=a[i];--j){
           f[j]|=f[j-a[i]];
       }
   }

   for(int j=m;j<M;++j) {
       if(f[j]){
           cout<<j;
           break;
       }
   }
   return 0;
}

// 202209-3 防疫大数据

// 思路：利用哈希表存储当前日期下每个地区风险的持续时间（开始->结束）
//在当前天数i下，遍历前6天的漫游信息，判断信息对应的日期和地区是否有风险且满足题意的三个条件即可，最后排序去重即可

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <unordered_map>

using namespace std;

const int N = 1010;
int n, r[N], m[N];
vector<int> v[N];       // 每天的答案

struct Data {
    int d, u, r;
};
vector<Data> vi[N];     // 存储每天的漫游信息

// [start,last] 地区最近的持续时间
unordered_map<int, int> last;
unordered_map<int, int> start;


int main() {

    cin >> n;

    int cnt = 0;
    for (int i = 0; i < n; i++) {
        cin >> r[i] >> m[i];		// 风险地区个数，信息个数
        int p;
        for (int j = 0; j < r[i]; j++) {
            cin >> p;           // 风险地区
            if (last.count(p) == 0 || last[p] + 1 < i) // 若不存在或过期，更新
                start[p] = i;
            last[p] = i + 6;        //截至（last）时间更新
        }
        for (int j = 0; j < m[i]; j++) {        // 漫游数据
            int di, ui, ri;
            cin >> di >> ui >> ri;		// 日期，用户，地区
            //地区无风险或者当前时间超过风险last时间 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // 需要满足上述存在风险的地区自到访日至生成名单当日持续处于风险状态。
            if (last.count(ri) == 0 || last[ri] < i) continue;
            vi[i].push_back({di,ui,ri});     // 存入对应天数的漫游信息下
        }
        // 前6天找
        for (int j = max(0, i - 6); j <= i; j++) {
            for (int t = 0; t < vi[j].size(); t++) {
                Data u = vi[j][t];
                // 当前漫游信息的日期~第i日：地区都是有风险 (上述存在风险的地区自到访日至生成名单当日持续处于风险状态。)
                //  名单生成日是风险 && 到访的时间在7天内 && 到访的时间处于风险时间！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                if (last[u.r] >= i && u.d >= i - 6 && u.d >= start[u.r] && u.d <= last[u.r])		// i>=u.d, 故后面 i>=u.d >= start[u.r]
                    v[i].push_back(u.u);
            }
        }
    }

    // 排序+去重 （注意一定要先排序）
    for (int i = 0; i < n; i++) {
        sort(v[i].begin(), v[i].end());
        v[i].erase(unique(v[i].begin(), v[i].end()), v[i].end());       // 去重！！！！！！！！！！！！！！！！！！！！
        cout << i;
        for (int user = 0; user < v[i].size(); user++) cout << ' ' << v[i][user];
        cout << '\n';
    }
    return 0;

}


// 202209-4 吉祥物投票
// 题意：有m个作品，n个人进行投票，有操作1，将l-r编号的人投给x作品，操作2,投给x作品的人投给w(w！=0)，操作3，投x的和投y的互换，操作4，查询投w的有多少人，操作5，当前作品谁能赢
// 思路：利用set为维护区间，使用set的二分进行区间的修改，使用并查集进行懒更新，使用set模拟优先队列找出最多的票数的作品（因为优先队列没有删除指定数据的功能）
// 具体的，设每个作品都下标和编号id，下标即作品在数组中的位置（不再改变），id也代表了作品（可变，同一个id可能一会代表了这个作品，一会又代表了另一个作品）
// p[x] 是x下标的作品的id,fa[x]是id所代表作品的id（实际上是并查集），map<int,int>cnt 映射了id所代表的作品的票数，map<int,int>mp,映射了id所代表的作品的下标
// 核心思想：通过id这个中间层，避免了很多修改
// set维护区间,并查集进行懒更新,set模拟优先队列找出最多的票数的作品
#include <bits/stdc++.h>
using namespace std;

const int N = 200010;
struct Seg {
    int l, r, id;
    bool operator<(const Seg& s)const {
        return l < s.l;
    }
};
struct Node {
    int id, cnt;
    bool operator<(const Node& a)const {
        return cnt != a.cnt ? cnt > a.cnt : id < a.id;      // ！！！！！！！！！！！！
    }
};
set<Seg> seg;
set<Node> heap;     // 按票数对作品排名，！！！！！！！！！！！！！
map<int, int> cnt;      // 每个作品票数，！！！！！！！！！！！！！！！
int p[N], fa[N];        // fa[x] 存储的是值x实际上维护的是那一种作品,也就是说seg中不同的值id可能代表同一个作品
int n, m, tot;
map<int, int> mp;       //映射回该作品的真正的ID，！！！！！！！！！！！！！！！！

int find(int x) { return x == fa[x] ? x : x = find(fa[x]); }
// x合并到y
void union_set(int x, int y) { fa[find(x)] = find(y); }


void Op1() {
    int l, r, x; cin >> l >> r >> x;
    // 左闭右开
    auto bg = seg.upper_bound({ l,-1,-1 });
    --bg;
    auto ed = seg.upper_bound({ r,-1,-1 });
    vector<Seg> add;        // !!!!!!!!!!!!!!!!!!!!!!

    // 删除所覆盖的区间的信息，每次都要先删除heap中旧的值，修改完再插入新的值！！！！！！！！！！！！！！！！！
    for (auto it = bg; it != seg.end() && it != ed; ++it) {
        int rid = find(it->id);         // !!!!!!!!!!!!!!!!! 每次都的先要find找到实际代表的id
        heap.erase({ rid,cnt[rid] });       // ！！！！！！！！！！！！！！
        cnt[rid] -= (it->r - it->l + 1);
        if (rid && cnt[rid] > 0) heap.insert({ rid,cnt[rid] });     // !!!!!!!!!!!!
    }
    if (bg->l < l) {
        int rid = find(bg->id);
        heap.erase({ rid,cnt[rid] });
        cnt[rid] += (l - bg->l);
        if (rid && cnt[rid] > 0) heap.insert({ rid,cnt[rid] });
        add.push_back({ bg->l,l - 1,rid });
    }
    --ed;       // !!!!!!!!!!!!
    if (ed->r > r) {
        int rid = find(ed->id);
        heap.erase({ rid,cnt[rid] });
        cnt[rid] += (ed->r - r);
        if (rid && cnt[rid] > 0) heap.insert({ rid,cnt[rid] });
        add.push_back({ r + 1,ed->r,rid });
    }
    seg.erase(bg, ++ed);
    for (auto i : add) seg.insert(i);
    int rid = p[x];
    seg.insert({ l,r,rid });
    heap.erase({ rid,cnt[rid] });
    cnt[rid] += (r - l + 1);
    heap.insert({ rid,cnt[rid] });
}
void Op2() {
    int x, w;
    cin >> x >> w;
    // 两个都先删，加完再插，合并 ！！！！！！！！！！！！！！
    heap.erase({ p[x], cnt[p[x]] });
    heap.erase({ p[w], cnt[p[w]] });
    cnt[p[w]] += cnt[p[x]];
    if (p[w])heap.insert({ p[w],cnt[p[w]] });
    union_set(p[x], p[w]);
    p[x] = ++tot;       // ！！！！！！！！！！！！！！！！
    mp[tot] = x;        // ！！！！！！！！！！！！！！！！

}
void Op3() {
    int x, y;
    cin >> x >> y;
    swap(mp[p[x]], mp[p[y]]);       // 下标互换，
    swap(p[x], p[y]);           // id互换
}
void Op4() {
    int w;
    cin >> w;
    cout << cnt[p[w]] << "\n";
}
void Op5() {
    if (heap.size() == 0) cout << 0 << "\n";
    else {

        int ct = 0, id = 0;     // ct得初始化为0，规定得票数至少为 1！！！！！！！！！！！！！！！！！！！！！！
        for (auto t : heap) {
            if (t.cnt > ct) {
                ct = t.cnt;
                id = mp[t.id];
            }
            if (t.cnt == ct && mp[t.id] < id) id = mp[t.id];
            if (ct > t.cnt) break;
        }
        cout << id << "\n";
    }
}
int main() {
    int T;
    cin >> n >> m >> T;
    tot = m;//!!!!!!!!!!!!!
    cnt[0] = n;     // 初始全为0
    seg.insert({ 1,n,0 });      // 初始全为0
    for (int i = 1; i <= m+T; ++i) fa[i] = p[i] = i;       // !!!!!!!!!!!!!!!!!!!!!!!!!!! 最多再产生T个id
    for (int i = 1; i <= m; ++i) mp[i] = i;

    while (T--) {
        int op;
        cin >> op;
        if (op == 1) Op1();
        else if (op == 2) Op2();
        else if (op == 3) Op3();
        else if (op == 4) Op4();
        else Op5();
    }
    return 0;
}


//202206-1

#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int n;
int a[1010];

int main()
{
	cin>>n;
	double sum=0;
	for(int i=1;i<=n;++i){
		cin>>a[i];
		sum+=a[i];
	}
	sum/=n;
	double fx=0;
	for(int i=1;i<=n;++i){
		fx+=(a[i]-sum)*(a[i]-sum);
	}
	fx/=n;
	for(int i=1;i<=n;++i){
		printf("%f\n",(a[i]-sum)/sqrt(fx));
	}
	
	return 0;
}


// 202206-2

#include <bits/stdc++.h>
using namespace std;
int n,L,S;
unordered_set<pair<int,int>> lp;
int sp[51][51];

int main()
{
	cin>>n>>L>>S;
	for(int i=1;i<=n;++i){
		int x,y;
		cin>>x>>y;
		lp.insert(make_pair(x,y));
	}

	int m=0;
	for(int i=S;i>=0;--i){
		for(int j=0;j<=S;++j){
			int t;
			cin>>sp[i][j];
		}
	}
	// 2500000
	int res=0;
	for(auto v:lp){
		int x=v.first;
		int y=v.second;
		
		if(x<=L-S && y<=L-S){
			bool ok=true;
			for(int i=0;i<=S;++i){
				for(int j=0;j<=S;++j){
					int nx=x+i,ny=y+j;
					if(sp[i][j]==1){
						if(lp.find(make_pair(nx,ny))==lp.end()) ok=false;
					}else{
						if(lp.find(make_pair(nx,ny))!=lp.end()) ok=false;
					}
					if(!ok) break;
				}
				if(!ok) break;
			}
  			if(ok) ++res;
		}
	}
	cout<<res<<endl;
	return 0;
}


// 	202206-3  	角色授权
#include <bits/stdc++.h>
using namespace std;

struct node {
    unordered_set<string> ops, category, resource;		// 操作，种类，资源
};
unordered_map<string, unordered_set<string>> user_to_role;      // 存储用户以及用户组对应的角色！！！！！！！！！！！！！！！！！！！！
unordered_map<string, node> role;       

int n, m, q;
string op, cate, resource_name;

void get_role(const vector<string>& arr,vector<string>& res) {
    for (auto i : arr) {
        for (auto j : user_to_role[i])
            res.push_back(j);
    }
}
// 按照题目来
bool check(string s) {
    auto& nd = role[s];
    if (nd.ops.count(op) == 0 && nd.ops.count("*")==0) return false;
    if (nd.category.count(cate) == 0 && nd.category.count("*")==0) return false;
    if (nd.resource.count(resource_name) == 0 && !nd.resource.empty()) return false;
    return true;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin >> n >> m >> q;
    for (int i = 0; i < n; ++i) {	// 读入每个角色的操作，种类，资源名称
        string role_name,str;
        int k;
        cin >> role_name >> k;
        for (int j = 0; j < k; ++j) cin >> str, role[role_name].ops.insert(str);
        cin >> k;
        for (int j = 0; j < k; ++j) cin >> str, role[role_name].category.insert(str);
        cin >> k;
        for (int j = 0; j < k; ++j)cin >> str, role[role_name].resource.insert(str);
    }
    for (int i = 0; i < m; ++i) {		// 读入与角色关联的用户和用户组
        string role_name,ug,ug_name; int ns;
        cin >> role_name >> ns;
        for (int j = 0; j < ns; ++j) {
            cin >> ug >> ug_name;
            user_to_role[ug_name].insert(role_name);
        }
    }
    for (int i = 0; i < q; ++i) {       
        string user_name, str; int ng;
        cin >> user_name >> ng;
        vector<string> arr;
        arr.push_back(user_name);
        for (int j = 0; j < ng; ++j) {
            cin >> str;
            arr.push_back(str);
        }
        vector<string>rs; 
        get_role(arr,rs);

        cin >> op >> cate >> resource_name;
        
        bool ok = false;
        for (auto& s : rs) {
            if (check(s)) {
                ok = true;
                break;
            }
        }
        cout << (ok ? 1 : 0) << "\n";
    }
    return 0;
}

// 202206-4 光线追踪
// 202206-4
// 思路：所有的光线传播路径都是与坐标平行，且所有光源与反射点都是正数，且反射线段的长度有限，不大于3*1e5可以进行离散化处理
// 使用两个map<int,map<int,int>>存储x轴相同的点（y坐标和反射面编号），且按y递增排序，存储y轴相同的点，且按x轴排序
// 随后进行光线追踪的模拟
#include<iostream>
#include<map>
#include<cmath>
using namespace std;

// 光线的方向!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!,根据题目d的含义来定义
enum class LightDir {
    X_POS = 0,
    Y_POS = 1,
    X_NEG = 2,
    Y_NEG = 3
};
// 左上右
int dx[] = { 1,0,-1,0 }, dy[] = { 0,1,0,-1 };

struct point {
    int x, y;
    bool operator==(const point& p) const {
        return x == p.x && y == p.y;
    }
};

typedef pair<point, int> point_with_int;        // ！！！！！！
const point NULL_POINT = { 2e9, 2e9 };
const point_with_int NULL_PD = { NULL_POINT, -1 };

struct mirror {
    int x1, y1, x2, y2, k, id;
    double a;
    void read(int id) {
        cin >> x1 >> y1 >> x2 >> y2 >> a;
        if (x1 > x2) {      // ！！！！！！！！！！！！！！
            swap(x1, x2); swap(y1, y2);
        }
        k = (y2 - y1) / (x2 - x1);      // 斜率，1或-1 !!!!!!!!!!!!!!!!!!!!!!
        this->id = id;
    }
}mirrors[200001];       // 每一个反射线段要存储起来，因为后面可能要对某个线段删除

class space {
    /*
    * psx = 所有镜面整点按x方向排序；
    * key存放所有点的y值，value递增存储对应x坐标和倾斜方向
    * psy = 所有镜面整点按y方向排序；
    * key存放所有点的x值，value递增存储对应y坐标和倾斜方向
    */
    map<int, map<int, int> > psx, psy;

    void add_point(int x, int y, int id) {
        auto it = psx.find(y);      // ！！！！！！！！！！！！！！！！
        if (it == psx.end()) {
            map<int, int> add;
            add.insert({ x,id }); psx.insert({ y, add });
        } else {
            it->second.insert({ x,id });
        }

        it = psy.find(x);
        if (it == psy.end()) {
            map<int, int> add;
            add.insert({ y,id }); psy.insert({ x, add });
        } else {
            it->second.insert({ y,id });
        }
    }

    void del_point(int x, int y) {
        psx[y].erase(x);
        psy[x].erase(y);
    }

public:
    void add_mirror(mirror m) {
        //保证光源不位于当前存在的某个反射面（不含端点）上
        for (int x = m.x1 + 1, y = m.y1 + m.k; x < m.x2; x++, y += m.k) {       // !!!!!!!! 注意范围
            add_point(x, y, m.id);
        }
    }
    void del_mirror(mirror m) {
        for (int x = m.x1 + 1, y = m.y1 + m.k; x < m.x2; x++, y += m.k) {
            del_point(x, y);
        }
    }
    //返回反射点和反射面编号！！！！！！！！！！！！！
    point_with_int find_nearst_reflect_point(point p, LightDir d) {
        if (d == LightDir::X_POS || d == LightDir::X_NEG) {     // x方向
            auto it = psx.find(p.y);
            if (it == psx.end()) return NULL_PD;
            map<int, int>::iterator it2;
            if (d == LightDir::X_POS) {
                it2 = it->second.upper_bound(p.x);      // upper_bound
                if (it2 == it->second.end()) return NULL_PD;
            } else {
                it2 = it->second.lower_bound(p.x);
                if (it2 == it->second.begin()) return NULL_PD;
                --it2;//技巧：lower_bound的前一个就是第一个比p.x小的数
            }
            return { {it2->first,p.y}, it2->second };
        } else {                                    // y方向
            auto it = psy.find(p.x);
            if (it == psy.end()) return NULL_PD;
            map<int, int>::iterator it2;
            if (d == LightDir::Y_POS) {
                it2 = it->second.upper_bound(p.y);
                if (it2 == it->second.end()) return NULL_PD;
            } else {
                it2 = it->second.lower_bound(p.y);
                if (it2 == it->second.begin()) return NULL_PD;
                --it2;
            }
            return { {p.x,it2->first}, it2->second };
        }
    }
}instance;

LightDir next_dir(LightDir dir, int mirror_k) {     // !!!!!!!!!!!!!!!!!!!!!!!
    if (dir == LightDir::X_POS) {
        return mirror_k == 1 ? LightDir::Y_POS : LightDir::Y_NEG;
    } else if (dir == LightDir::X_NEG) {
        return mirror_k == -1 ? LightDir::Y_POS : LightDir::Y_NEG;
    } else if (dir == LightDir::Y_POS) {
        return mirror_k == 1 ? LightDir::X_POS : LightDir::X_NEG;
    } else {//(dir == LightDir::Y_NEG)
        return mirror_k == -1 ? LightDir::X_POS : LightDir::X_NEG;
    }
}
pair<point, int> test_source(point p, LightDir d, double I, int T) {
    while (T > 0) {
        point_with_int ret = instance.find_nearst_reflect_point(p, d);
        point p2 = ret.first; int id = ret.second;
        int dist = abs(p.x - p2.x) + abs(p.y - p2.y);       // ！！！！！！！！！！！
        if (p2 == NULL_POINT || dist > T) {
            p.x += dx[(int)d] * T;      // 更新最终点的坐标
            p.y += dy[(int)d] * T;      // ！！！！！！！！！！！！！！！！！
            break;
        }
        p = p2;
        d = next_dir(d, mirrors[id].k);
        I = I * mirrors[id].a;
        if (I < 1.0) {
            return { {0,0},0 };
        }
        T -= dist;
    }
    return { p,(int)I };
}

int main() {
    ios::sync_with_stdio(false);
    int m;
    cin >> m;
    for (int i = 1; i <= m; i++) {
        int op; cin >> op;
        if (op == 1) {
            mirrors[i].read(i);
            instance.add_mirror(mirrors[i]);
        } else if (op == 2) {
            int k; cin >> k;
            instance.del_mirror(mirrors[k]);
        } else {
            int x, y, d, t;
            double I;
            cin >> x >> y >> d >> I >> t;
            auto ans = test_source({ x,y }, (LightDir)d, I, t);
            cout << ans.first.x << ' ' << ans.first.y << ' ' << ans.second << endl;
        }
    }
    return 0;
}


//202203-1
#include <bits/stdc++.h>
using namespace std;

const int N=100010;
int n,k;
bool vis[N];
	

int main()
{
	cin>>n>>k;
	vis[0]=true;
	int res=0;
	for(int i=1;i<=k;++i){
		int x,y;
		cin>>x>>y;
		if(!vis[y]) ++res;
		vis[x]=true;
	}
	cout<<res<<endl;
	
	
	return 0;
}


//202203-2

#include <bits/stdc++.h>
using namespace std;

const int N=200010;
int n,m,k;

int diff[N];

int main()
{
	cin>>n>>m>>k;
	int mx=0;
	for(int i=1;i<=n;++i){
		int in,c;
		cin>>in>>c;
		// 打疫苗区间[in-c-k+1,in-k]
		// t+k<=in<=t+k+c-1 => t<=in-k t>=in-c-k+1
		diff[max(1,in-c-k+1)]+=1;
		diff[max(1,in-k+1)]-=1;
		mx=max(mx,in-k+1);
	}
	for(int i=1;i<=mx;++i) {
		diff[i]+=diff[i-1];
	}
	
	for(int i=1;i<=m;++i){
		int q;
		cin>>q;
		cout<<diff[q]<<endl;
	}

	return 0;
}


// 202203-3 计算资源调度器

// 思路，用set存储每个可用区的计算节点，用set存储计算节点里的任务，l[N]记录计算节点属于那个可用区
// f累计只有2000，故对于每一个f，遍历所有计算节点，同时进行na和paa的处理，随后对于节点i所处的可用区进行pa任务的亲和
#include <bits/stdc++.h>

using namespace std;
const int INF = 1e9;
int n, m;
int cnt[1010], l[1010];		// cnt[x] 是计算节点x上的任务个数
set<int> ss[1010];//可用区i里的节点
set<int> s[1010];//节点i里的任务
// na是要运行在编号为na的可用区内；
// pa是必须运行在和编号为pa的应用的计算任务的可用区；
// paa是不能和编号为paa的应用的计算任务在同一个 计算节点 上运行；
int find(int na, int paa, int pa)//可用区、反亲和性、亲和性
{
   int Min = INF;      // ！！！！！！！！！！！
   int ans = 0;        // ！！！！！！！！！！！！！！！！
   for (int i = 1; i <= n; i++) {      // 从小到大，遍历n个计算节点，就不用在特殊判断编号小的计算节点了
       if (cnt[i] < Min && (l[i] == na || na == 0) && (s[i].find(paa) == s[i].end() || paa == 0))//判断可用区和反亲和
       {
           //判断任务亲和
           int p = l[i];
           int ok = 0;
           for (auto t : ss[p])        // ！！！！！！！！！！ 该可用区是否有亲和的节点
               if (s[t].find(pa) != s[t].end())ok = 1;

           if (ok || pa == 0) {
               Min = cnt[i];
               ans = i;
           }
       }
   }
   cnt[ans] ++;		// 该计算节点任务++		！！！！！！！！！
   return ans;
}
int main() {
   ios::sync_with_stdio(false);
   cin.tie(0), cout.tie(0);
   cin >> n >> m;
   for (int i = 1; i <= n; i++) cin >> l[i], ss[l[i]].insert(i);
   int g;
   cin >> g;
   for (int i = 1; i <= g; i++) {
       int f, a, na, pa, paa, paar;		
       cin >> f >> a >> na >> pa >> paa >> paar;// 个数，任务编号，节点亲和性，任务亲和性，任务反亲和性，尽量满足
       vector<int> ans;
       while (f--) {		// 一个一个来
           int x = find(na, paa, pa);
           if (x == 0 && paar == 0)//如果没有，看一下是不是必须满足，直接把反亲和传0就可以， ！！！！！！！！！！！
               x = find(na, 0, pa);
           s[x].insert(a);
           ans.push_back(x);
       }
       for (int j = 0; j < ans.size(); j++) cout << ans[j] << " ";
       cout << '\n';
   }
   return 0;
}


//  202203-4 通信系统管理
// 题意：n台计算机，每两台之间有计算额度，表明每天可互相发送的数据量，有操作u,v,x,y,为机器u和v的每日可用额度增大x,持续y天，每台机器“通信主要对象”：当前时刻与互联机器的额度最大的机器，通信对：互为“通信主要对象”，通信孤岛：该机器与任何一台机器的额度都为0
// 思路：按题意，一天天处理，增加可以看作是激活，过期看作是反激活，使用vector <info> deActive[maxn]存储每天的过期信息，使用map <pair<int, int>, ll> save 存储机器之间的额度
#include <bits/stdc++.h>

typedef long long ll;
const int maxn = 100010;
const int inf = ~(1u << 31u);
const ll linf = ~(1llu << 63u);

using namespace std;

struct node {
    ll value;
    int to;
    node(ll value, int to) : value(value), to(to) {}
    bool operator < (const node& d) const {
        return value == d.value ? to < d.to : value > d.value;      // !!!!!!! 额度相等，编号小的在前
    }
};

struct info {
    int u, v, x;
    info(int u, int v, int x) : u(u), v(v), x(x) {}
};

set <node> d[maxn];         // 利用set对邻居节点的额度进行排序
map <pair<int, int>, ll> save;      // 存储边的权值

vector <info> deActive[maxn];
int p_value, q_value;

// 检查一个点 是否为孤岛
int check_p(int x) {        // ！！！！！！！！！！！！
    return d[x].empty() || d[x].begin()->value == 0;
}

// 检查一个点 是否包含一个通讯对
int check_q(int x) {
    if (check_p(x)) return 0;   
    int y = d[x].begin()->to;
    return !check_p(y) && d[y].begin()->to == x;     //!!!!!!!!!!!  !check_p(y)是为了避免额度为0的情况
}

void work(int u, int v, int x) {
    // 因为要对额度做修改，需要先删除u点上的信息，随后在计算更新后的u点上的信息
    // 处理 孤岛数量 通讯对数量
    p_value -= check_p(u);      // ！！！！！！！！！！！！！
    q_value -= check_q(u);      // ！！！！！！！！！！！

    ll orgValue = save[{u, v}];         // ！！！！！！！
    save[{u, v}] += x;

    // 删除 旧的
    node org(orgValue, v);
    d[u].erase(org);
    // 插入新的
    d[u].emplace(save[{u, v}], v);
    
    // 处理 孤岛数量 通讯对数量
    p_value += check_p(u);
    q_value += check_q(u);
}

int main() {
    ios::sync_with_stdio(false);
    int n, m;
    cin >> n >> m;

    p_value = n;        // ！！！！！！！！！！！！！
    q_value = 0;        // ！！！！！！！！！！

    for (int i = 1; i <= m; ++i) {

        // 先处理过期的额度
        for (const auto& x : deActive[i]) {
            work(x.u, x.v, -x.x);
            work(x.v, x.u, -x.x);
        }

        int k;
        cin >> k;
        // 输入当天的额度
        for (int j = 1; j <= k; ++j) {
            int u, v, x, y;
            cin >> u >> v >> x >> y;
            if (i + y <= m) // ！！！！！！！！！！！！！！！ i+y
                deActive[i + y].emplace_back(u, v, x);//反向激活，第i+y天过期
            work(u, v, x);
            work(v, u, x);
        }

        int l;
        cin >> l;
        // 查询主要通讯对象
        for (int j = 1; j <= l; ++j) {
            int x;
            cin >> x;
            if (check_p(x)) cout << 0 << "\n";
            else cout << d[x].begin()->to << "\n";
        }

        int p, q;
        cin >> p >> q;
        // 孤岛数量
        if (p) cout << p_value << "\n";//换行用\n，否则刷新缓冲区会慢
        // 通讯对数量
        if (q) cout << q_value << "\n";
    }
    return 0;
}



// 202112-1
#include <bits/stdc++.h>
using namespace std;
int n,N;

int main()
{
	cin>>n>>N;
	int res=0,p=0;
	int a;
	// 0 2 5 8

	for(int i=1;i<=n;++i,p=a) {
	
		cin>>a;
		res+=(i-1)*(a-p);   // 由于a[i]严格单调增加，故f(j)=i-1,a[i-1]<=j<a[i]
	}
	cout<<res+n*(N-p)<<endl;
	return 0;
}


// 202112-2
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
int n,N,r;
int a[100010];

LL g(int x)
{
   return x/r;
}
int main()
{
	cin>>n>>N;
	r=N/(n+1);

   for(int i=1;i<=n;++i)
		cin>>a[i];

	a[n+1]=N;

	LL res=0;
	for(int i=1;i<=n+1;++i){
		  int left = a[i-1];
       int right = a[i] - 1;   //  注意这里不要和r重名了
       int value = i-1;
       int gap = 0;
       // 由于a[i]严格单调增加，故f(j)=i-1,其中j的范围位为a[i-1]<=j<a[i]
       for(int j = left;j<=right;j =j + gap)//在f(x)均相同的区间里，再对g(x)进行讨论
       {
           // g(x)=(n+1)*x/N=x/r
           int temp = (g(j)+1)*r-1;//temp是取值为g(j)最大的数
           if(temp > right) temp=right;//如果该数超过区间的范围了，则把right赋值给temp，调整边界
           gap = temp - j + 1;//在该gap的长度内，g(x),f(x)各自的值都相同，故可以同一结算
           res += (abs(value-g(j))*gap);
       }

	}
	cout<<res<<endl;
	return 0;

}


// 202112-3 登机牌条码 

#include <bits/stdc++.h> 

using namespace std;
const int N = 2000, mod = 929;	// 一个码字两个数，给2000大小够了！！！！！！！！！！！！

enum { U, L, D}state;

int w, s, k;
string str;
int d[N], g[N], res[N], cnt;
void inline push(int x) { d[++cnt] = x; }
int main() {
   cin >> w >> s >> str;
   k = ~s ? 1 << s + 1 : 0;

   // d //
   state = U;
   for (auto c : str) {
       if (isupper(c)) {
           if (state == L) push(28), push(28);
           else if (state == D) push(28);
           state = U;
           push(c - 'A');
       } else if (islower(c)) {
           if (state != L) push(27);
           state = L;
           push(c - 'a');
       } else {
           if (state != D) push(28);
           state = D;
           push(c - '0');
       }
   }
   if (cnt & 1) push(29);        // ！！！！！！！！！！！！！！
   for (int i = 1, j = 1; i < cnt; i += 2, ++j)
       d[j] = 30 * d[i] + d[i + 1];
   cnt >>= 1;        // ！！！！！！！！！！！！
   while ((cnt + 1 + k) % w) push(900);      // ！！！！！！！！！！！！！
   d[0] = cnt+1;         // ！！！！！！！！！！！！

   // g //
   g[0] = 1;       // 最高次系数为1      // ！！！！！！！！！！！！！！
   int r = -3;       // ！！！！！！！！！！
   for (int i = 1; i <= k; r = r * 3 % mod, ++i) {     // 递推计算多项式
       for (int j = i - 1; j >= 0; --j)        // 最高项不用动
           g[j + 1] = (g[j + 1] + g[j] * r) % mod;
   }
   //x^k d(x)≡q(x)g(x)-r(x);
   // 为了消除q(x)对计算r(x)的干扰，在恒等式两边同时对g(x)取余，则公式转换成x^k d(x) mod g(x) ≡ -r(x) mod g(x);
   // 问题转换成求x^k d(x) mod g(x)；，最后对该式取反即可，因为r(x)是不超过k-1次的多项式，而g（x)是k次多项式，故右边还是-r(x)
   // r //

   for (int i = 0; i <= cnt; ++i) res[i] = d[i];
   for (int i = 0; i <= cnt; ++i) {
       int R = res[i];
       res[i] = 0;       // ！！！！！！！！！！！
       for (int j = 1; j <= k; ++j) {      // 模拟除法，g的最高次系数为1，乘d的最高此项次数进行消去，从1开始
           res[i + j] = (res[i + j] - R * g[j]) % mod;
       }
   }
   for (int i = 0; i <= cnt; ++i) cout << d[i] << '\n';
   for (int i = cnt + 1; i <= cnt + k; ++i) cout << (-res[i] % mod + mod) % mod << '\n';
   return 0;
}


//202112-4	磁盘文件操作
// 原文链接：https://blog.csdn.net/Crispo_W/article/details/123512092

#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
using namespace std;

#define FREE 0
#define OCCUPIED 1
#define ZOMBIE 2

#define N_MAX 200005
#define MIXED 2000000020
#define FULL (-2000000020)

struct FILE_BLOCK {
    int l, r;
    int x;  // value: -1e9~1e9, 2e9: mixed
    int id; // id: 1~n(<2e5), 2e9: mixed
    int state;  // FREE, OCCUPIED, ZOMBIE, 2e9: mixed
};
struct OPRT {
    OPRT(int t, int i, int ll, int rr, int xx, int pp) : type(t), id(i), l(ll), r(rr), x(xx), p(pp) {}
    int type, id, l, r, x, p;
};

int n, m, k;
vector<FILE_BLOCK> fb(N_MAX << 4);
vector<OPRT> oprt;
vector<int> coordinates;

int discretization() {
    coordinates.push_back(0);   // let coordinates begin with 1
    sort(coordinates.begin(), coordinates.end());
    m = unique(coordinates.begin(), coordinates.end()) - coordinates.begin();   // 1~m
    coordinates.resize(m);
    for (auto & op : oprt) {
        switch (op.type) {
            case 0: case 1: case 2:
                op.l = lower_bound(coordinates.begin(), coordinates.end(), op.l) - coordinates.begin();
                op.r = lower_bound(coordinates.begin(), coordinates.end(), op.r) - coordinates.begin();
                break;
            case 3: default:
                op.p = lower_bound(coordinates.begin(), coordinates.end(), op.p) - coordinates.begin();
                break;
        }
    }
    return 0;
}

void pushup(int curr) {
    if (fb[curr<<1].id == fb[curr<<1|1].id) fb[curr].id = fb[curr<<1].id;
    else fb[curr].id = MIXED;
    if (fb[curr<<1].x == fb[curr<<1|1].x) fb[curr].x = fb[curr<<1].x;
    else fb[curr].x = MIXED;
    if (fb[curr<<1].state == fb[curr<<1|1].state) fb[curr].state = fb[curr<<1].state;
    else fb[curr].state = MIXED;
}
/* f5 */
void build(int curr, int l, int r){
    fb[curr].l = l;
    fb[curr].r = r;
    if (l == r) return;
    else {
        int mid = l+((r-l)>>1);     // 用减法不用加法，避免爆int
        build(curr<<1, l, mid);
        build((curr<<1)+1, mid+1, r);
        pushup(curr);
    }
}
void pushdown(int curr) {
    if (fb[curr].r != fb[curr].l) {
        if(fb[curr].id != MIXED) fb[curr << 1].id = fb[curr << 1 | 1].id = fb[curr].id;
        if(fb[curr].state != MIXED) fb[curr << 1].state = fb[curr << 1 | 1].state = fb[curr].state;
        if(fb[curr].x != MIXED) fb[curr << 1].x = fb[curr << 1 | 1].x = fb[curr].x;
    }
}
/* Interval change */
/* update_test: Find the rightmost available to_r */
int update_test(int curr, int to_l, int id) {
    if (fb[curr].state == FREE || fb[curr].state == ZOMBIE || fb[curr].id == id) return fb[curr].r;
    else if (fb[curr].state == OCCUPIED && fb[curr].id != MIXED) return FULL;   // debug: state = mixed, maybe itself
    else {  // state == mixed
        int mid = fb[curr].l+((fb[curr].r-fb[curr].l)>>1);
        pushdown(curr);
        if (to_l <= mid) {
            int left_r = update_test(curr<<1, to_l, id);
            if (left_r < mid) return left_r;
            int right_r = update_test(curr<<1|1, to_l, id);
            return right_r == FULL ? left_r : right_r;
        }
        else {
            return update_test(curr<<1|1, to_l, id);
        }
    }
};
void update_ic(int curr, int to_l, int to_r, int x, int id) {
    if (fb[curr].l == to_l && fb[curr].r == to_r) { // 刚好覆盖
        fb[curr].state = OCCUPIED;
        fb[curr].id = id;
        fb[curr].x = x;
        return;
    }
    int mid = fb[curr].l+((fb[curr].r-fb[curr].l)>>1);
    pushdown(curr);
    if (to_l <= mid) update_ic(curr<<1, to_l, min(to_r, mid), x, id);
    if (to_r > mid) update_ic(curr<<1|1, max(to_l, mid+1), to_r, x, id);
    pushup(curr);
};
/* Interval deletion */
bool deletion_test (int curr, int to_l, int to_r, int id) {
    if (fb[curr].l == to_l && fb[curr].r == to_r) { // 只在完全匹配的时候作判断
        if (fb[curr].state == OCCUPIED && fb[curr].id == id) return true;
        else return false;
    }
    int mid = fb[curr].l+((fb[curr].r-fb[curr].l)>>1);
    pushdown(curr);
    bool avai = true;
    if (to_l <= mid) avai &= deletion_test(curr<<1, to_l, min(to_r, mid), id);
    if (to_r > mid) avai &= deletion_test(curr<<1|1, max(to_l, mid+1), to_r, id);
    return avai;
}
void deletion (int curr, int to_l, int to_r, int id) {
    if (fb[curr].l == to_l && fb[curr].r == to_r) {
        fb[curr].state = ZOMBIE;
        return;
    }
    int mid = fb[curr].l+((fb[curr].r-fb[curr].l)>>1);
    pushdown(curr);
    if (to_l <= mid) deletion(curr<<1, to_l, min(to_r, mid), id);
    if (to_r > mid) deletion(curr<<1|1, max(to_l, mid+1), to_r, id);
    pushup(curr);
}
/* Interval recovery */
bool recover_test (int curr, int to_l, int to_r, int id) {
    if (fb[curr].l == to_l && fb[curr].r == to_r) {
        if (fb[curr].state == ZOMBIE && fb[curr].id == id) return true;
        else return false;
    }
    int mid = fb[curr].l+((fb[curr].r-fb[curr].l)>>1);
    pushdown(curr);
    bool avai = true;
    if (to_l <= mid) avai &= recover_test(curr<<1, to_l, min(to_r, mid), id);
    if (to_r > mid) avai &= recover_test(curr<<1|1, max(to_l, mid+1), to_r, id);
    return avai;
}
void recover (int curr, int to_l, int to_r, int id) {
    if (fb[curr].l == to_l && fb[curr].r == to_r) {
        fb[curr].state = OCCUPIED;
        return;
    }
    int mid = fb[curr].l+((fb[curr].r-fb[curr].l)>>1);
    pushdown(curr);
    if (to_l <= mid) recover(curr<<1, to_l, min(to_r, mid), id);
    if (to_r > mid) recover(curr<<1|1, max(to_l, mid+1), to_r, id);
    pushup(curr);
}
/* query */
int query (int curr, int p) {
    if (fb[curr].l == fb[curr].r) return curr;
    else if (fb[curr].state != MIXED && fb[curr].id != MIXED && fb[curr].x != MIXED) return curr;
    else {
        int mid = fb[curr].l+((fb[curr].r-fb[curr].l)>>1);
        pushdown(curr);
        if (p <= mid) return query(curr<<1, p);
        else return query(curr<<1|1, p);
    }
}

int main() {
    scanf("%d%d%d", &n, &m, &k);
    int type=0, id=0, l=0, r=0, x=0, p=0;
    for (int i = 0; i < k; i++) {
        scanf("%d", &type);
        switch (type) {
            case 0:
                scanf("%d%d%d%d", &id, &l, &r, &x);
                oprt.emplace_back(type, id, l, r, x, 0);
                break;
            case 1:
                scanf("%d%d%d", &id, &l, &r);
                oprt.emplace_back(type, id, l, r, 0, 0);
                break;
            case 2:
                scanf("%d%d%d", &id, &l, &r);
                oprt.emplace_back(type, id, l, r, 0, 0);
                break;
            case 3:
                scanf("%d", &p);
                oprt.emplace_back(type, 0, 0, 0, 0, p);
                break;
            default:
                break;
        }
        if (type == 0 || type == 1 || type == 2) {
            coordinates.push_back(l);
            coordinates.push_back(r);
            if (l != 1) coordinates.push_back(l-1);     // 注意离散化要加入前后的点
            if (r != m) coordinates.push_back(r+1);
        }
        else {
            coordinates.push_back(p);
        }
    }
    /* 离散化 */
    discretization();
    /* 建树 */
    build(1, 1, m-1);
    int query_index;
    for (auto & op : oprt) {
        switch (op.type) {
            case 0:
                op.r = min(op.r, update_test(1, op.l, op.id));
                if (op.r != FULL) update_ic(1, op.l, op.r, op.x, op.id);
                printf("%d\n", op.r==FULL?-1:coordinates[op.r]);
                break;
            case 1:
                if (deletion_test(1, op.l, op.r, op.id)) {
                    deletion(1, op.l, op.r, op.id);
                    printf("OK\n");
                }
                else {
                    printf("FAIL\n");
                }
                break;
            case 2:
                if (recover_test(1, op.l, op.r, op.id)) {
                    recover(1, op.l, op.r, op.id);
                    printf("OK\n");
                }
                else printf("FAIL\n");
                break;
            case 3:
                query_index = query(1, op.p);
                if (fb[query_index].state != OCCUPIED) printf("%d %d\n", 0, 0);
                else printf("%d %d\n", fb[query_index].id, fb[query_index].x);
                break;
            default:
                break;
        }
    }
    return 0;
}



//202109-1
#include <bits/stdc++.h>
using namespace std;
int B[110];

int n;

int main()
{
	cin>>n;
	for(int i=1;i<=n;++i) cin>>B[i];
	
	int mx=INT_MIN,mi=INT_MAX;
	int sum_max=0,sum_min=0;
	for(int i=1;i<=n;++i){
		if(B[i]>mx){
			mx=B[i];
			sum_max+=B[i];
			sum_min+=B[i];
		}else{
			sum_max+=mx;
		}
	}
	cout<<sum_max<<endl;
	cout<<sum_min<<endl;
	return 0;
}


// 202109-2

#include <bits/stdc++.h>
using namespace std;
const int N = 500010;
const int M = 10010;
int a[N], d[M];
int n;
// 类似岛屿问题，假设一开始p非常大，水面淹没了所有岛屿，随着p的减小，即水面的下降，
// 岛屿数量出现变化。每当一个凸峰出现，岛屿数就会多一个；
// 每当一个凹谷出现，原本相邻的两个岛屿就被这个凹谷连在一起了，岛屿数减少一个。

// 使用差分数组 d[i] 统计每个非零元素作为分界点时的贡献，即当p=i时的差分数
// 如果它是局部最大值（前一个元素小于它且后一个元素小于它），则 d[a[i]]++，
// 如果是局部最小值（前一个元素大于它且后一个元素大于它），则 d[a[i]]--。

int main()
{
	cin>>n;
	for(int i=1;i<=n;++i) cin>>a[i];
	a[0]=a[n+1]=0;

	// uniqur 返回的是去重后的右边界（闭区间）
	n=unique(a,a+n+2)-a-1;
	// 此时[1,n)上即为去重后的元素，且a[0]=a[n]=0
	

	for(int i=1;i<n;++i){
		if(a[i-1]<a[i] && a[i]>a[i+1])  // 凸峰
			d[a[i]]--;
		else if(a[i-1]>a[i] && a[i]<a[i+1]) // 凹峰
			d[a[i]]++;
	}

	int res=0,sum=1;
	for(int i=0;i<M;++i)     // 水平面下降
		sum+=d[i],res=max(res,sum);
	cout<<res<<endl;

	return 0;
}

// 法二
// 想象一个跌宕起伏的山群，遍历a[i]时就像是扫描高低不同的山峰，对于一个山峰，我们直选哟
#include <bits/stdc++.h>
using namespace std;
const int N = 500010;
const int M = 10010;
int a[N], diff[M];      // diff 差分数组
int n;

int main()
{
	cin>>n;
	for(int i=1;i<=n;++i) cin>>a[i];
	
	for(int i=1;i<=n;++i){
		if(a[i-1]<a[i]){
			// 当p在迎风坡 [ a[i-1]+1.a[i] ]上水面穿过，这个山坡的贡献+1
			diff[a[i-1]+1]++,diff[a[i]+1]--;
		}
	}
	int res=0,sum=0;
	for(int i=1;i<M;++i){
		sum+=diff[i];
		res=max(res,sum);
	}
	
	cout<<res<<endl;
	
	return 0;
}


#include <bits/stdc++.h>
using namespace std;

const int N=500010;
const int M=10010;

int n;
int book[N];	// 记录a[i]是否露出水面,初始时全为0，全都露出水面
vector<int> v[M];

int main()
{
	cin>>n;
	int a;
	for(int i=1;i<=n;++i){
		cin>>a;
		v[a].push_back(i);
	}

	int res=0;
	if(v[0].size()!=n){
		book[0]=book[n+1]=1;	// 人为设置边界
		int last=1;	// 上一个p（水平面）时的岛屿数，开始时，水平面最低，整个为一个山峰

		for(int i=0;i<M;++i){	// 水平面上升
			if(!v[i].empty()){
				int t=last;
				for(int j=0;j<v[i].size();++j){
					book[v[i][j]]=1;	// 水面之下
                   // 若两则山峰均在水面之上，则山峰+1
                   // 若两侧山峰均在水面之下，则山峰-1
					if(book[v[i][j]-1] && book[v[i][j]+1]) --t;
					else if(book[v[i][j]-1]==0 && book[v[i][j]+1]==0) ++t;
				}
				res=max(res,t);
				last=t;
			}
		}
	}
	cout<<res<<endl;
	return 0;
}


// 	202109-3 脉冲神经网络
#include <bits/stdc++.h>

// 神经元：按照一定的公式 更新内部状态 ， 接受脉冲 并可以 发放脉冲
// 脉冲源：在特定的时间发放脉冲
// 突触：连接神经元-神经元或者脉冲源-神经元，负责传递脉冲

using namespace std;
typedef long long LL;
//1000个脉冲源+1000个神经源
const int N = 2005, INF = 0x3f3f3f3f;

//表示一共有 N 个神经元，S 个突触和 P 个脉冲源，输出时间刻 T
int n, s, p, T;
//一个正实数 Δt，表示时间间隔
double dt;
int h[N], e[N], D[N], ne[N], idx;
double w[N], v[N], u[N], a[N], b[N], c[N], d[N];
int r[N];
//存整个过程中神经元发脉冲的次数
int cnt[N];
//存某时刻某个神经元的内部参数，I[i][j],为i时刻j神经元的强度
double I[1024][N / 2];

static unsigned long _next = 1;

/* RAND_MAX assumed to be 32767 */
//这里要吧所给函数的next换一个变量名
int myrand(void) {
   _next = _next * 1103515245 + 12345;
   return((unsigned)(_next / 65536) % 32768);
}

void add(int aa, int bb, double cc, int dd) {
   e[idx] = bb;
   w[idx] = cc;
   D[idx] = dd;
   ne[idx] = h[aa];
   h[aa] = idx++;
}

int main() {
   scanf("%d%d%d%d", &n, &s, &p, &T);
   scanf("%lf", &dt);
   memset(h, -1, sizeof h);
   //保证所有的 RN 加起来等于 N
   for (int i = 0; i < n;) {
       //每行有以空格分隔的一个正整数 RN 和六个实数 v u a b c d
       //rn表示下面要输入rn个神经元
       int rn;
       scanf("%d", &rn);
       double vv, uu, aa, bb, cc, dd;
       scanf("%lf%lf%lf%lf%lf%lf", &vv, &uu, &aa, &bb, &cc, &dd);
       //按顺序每一行对应 RN 个具有相同初始状态和常量的神经元
       for (int j = 0; j < rn; j++, i++) {
           v[i] = vv, u[i] = uu, a[i] = aa, b[i] = bb, c[i] = cc, d[i] = dd;
       }
   }
   //输入接下来的 P 行，每行是一个正整数 r，按顺序每一行对应一个脉冲源的 r 参数
   for (int i = n; i <= n + p - 1; i++) {
       cin >> r[i];
   }
   //循环数组的长度
   int mod = 0;
   //建图
   for (int i = 0; i < s; i++) {
       //其中 s 和 t 分别是入结点和出结点的编号；w 和 D 分别表示脉冲强度和传播延迟
       int ss, tt, dd;
       double ww;
       cin >> ss >> tt >> ww >> dd;
       add(ss, tt, ww, dd);
       mod = max(mod, dd + 1);     //
   }
   for (int i = 0; i < T; i++) {		// 枚举每个时刻
       //求出在循环数组中映射的坐标
       int t = i % mod;
       //遍历所有脉冲源		// ！！！！！！！！！！！
       for (int j = n; j <= n + p - 1; j++) {
           //脉冲源在每个时刻以一定的概率发放一个脉冲
           if (r[j] > myrand()) {
               //计算状态
               for (int k = h[j]; ~k; k = ne[k]) {
                   int x = e[k];
                   //更新每个点的Ik,当前这个点会像下一个点隔D[k]时间后发送脉冲
                   I[(t + D[k]) % mod][x] += w[k];
               }
           }
       }

       //枚举所有神经元		// ！！！！！！！！！！！
       for (int j = 0; j < n; j++) {
           double vv = v[j], uu = u[j];
           //根据公式，跟新状态
           v[j] = vv + dt * (0.04 * vv * vv + 5 * vv + 140 - uu) + I[t][j];
           u[j] = uu + dt * a[j] * (b[j] * vv - uu);
           //如果满足 vk≥30，神经元会发放一个脉冲
           if (v[j] >= 30) {
               for (int k = h[j]; ~k; k = ne[k]) {
                   int x = e[k];
                   //更新每个点的Ik,当前这个点会像下一个点隔D[k]时间后发送脉冲
                   I[(t + D[k]) % mod][x] += w[k];
               }
               //统计该点发脉冲的次数
               cnt[j]++;
               //同时，vk 设为 c 并且 uk 设为 uk+d，其中 c 和 d 也是常量。
               v[j] = c[j], u[j] += d[j];

           }
       }
       //因为是循环数组，所以用完一次一定要记得清空
       memset(I[t], 0, sizeof I[t]);
   }
   double maxv = -INF, minv = INF;
   int maxc = -INF, minc = INF;
   for (int i = 0; i < n; i++) {
       minv = min(minv, v[i]);
       maxv = max(maxv, v[i]);
       minc = min(minc, cnt[i]);
       maxc = max(maxc, cnt[i]);
   }
   printf("%.3lf %.3lf\n", minv, maxv);
   cout << minc << ' ' << maxc << endl;


}


// 202109-4  收集卡牌 
#include <bits/stdc++.h>

using namespace std;
const int N = 17;
double p[N], f[81][1 << 16];        // f[i][j]为目前有i个金币，卡牌的状态为j的情况下，期望抽卡的次数
int n, k;
double dfs(int depth,int coin, int state,int cnt) {     // 深度即为抽卡的次数
    if (f[coin][state]) return f[coin][state];
    if (cnt * k <= coin) return depth;              // ！！！！！！！！注意返回值的不同，上面的是记忆值
    double s = 0;
    for (int i = 0; i < n; ++i) {
        if ((state >> i) & 1)
            s += p[i] * dfs(depth + 1, coin + 1, state, cnt);   // 由期望的定义，这里直接用p[i]乘上下一层的期望既可，次数的话最后一层已经乘上了
        else
            s += p[i] * dfs(depth + 1, coin, state | (1 << i), cnt - 1);
    }
    return f[coin][state] = s;          // ！！！！！！！！！！！！！
}
int main() {
    cin >> n >> k;
    for (int i = 0; i < n; ++i) cin >> p[i];
    printf("%.10lf", dfs(0,0,0,n));
    return 0;
}




// 202104-1
#include <bits/stdc++.h>
using namespace std;

int n,m,L;
int h[257];
int main()
{
	cin>>n>>m>>L;
	int a;
	for(int i=0;i<n;++i)
		for(int j=0;j<m;++j)
		{
			cin>>a;
			++h[a];
		}
	for(int i=0;i<L;++i)
		cout<<h[i]<<' ';

	return 0;
}


// 202104-2
#include <bits/stdc++.h>
using namespace std;
const int N=610;
int n,L,r,t;
int sum[N][N];
double f(int i,int j){      // 题目虽然给的整数，但是平均值是浮点数
	int x1=max(1,i-r),y1=max(1,j-r);
	int x2=min(n,i+r),y2=min(n,j+r);
	int S = sum[x2][y2]-sum[x1-1][y2]-sum[x2][y1-1]+sum[x1-1][y1-1];
	return S/(double)((x2-x1+1)*(y2-y1+1));
}
int main()
{
	cin>>n>>L>>r>>t;

	for(int i=1;i<=n;++i)
		for(int j=1;j<=n;++j){
			cin>>sum[i][j];
			sum[i][j]+=sum[i-1][j]+sum[i][j-1]-sum[i-1][j-1];
		}
		
	int res=0;
	
	for(int i=1;i<=n;++i)
		for(int j=1;j<=n;++j){
			if(f(i,j)<=t)
				++res;
		}

	cout<<res<<endl;
	return 0;
}

// 202104-3 DHCP服务器

//// 跟着题目”实现细节“部分来照着做就行

#include <bits/stdc++.h>
using namespace std;

const int N = 10010;//所以时间复杂度应不超过n方

int n, m, t_def, t_max, t_min;//ip地址个数， 请求个数，分配个客户端的ip地址的默认过期时间长度
//最长和最短过期时间
string h;//主机名称
struct IP {
   int state; // 0:未分配， 1：待分配 2：占用 3：过期
   int t;//过期时间
   string owner;
}ip[N];     // 地址池

void update_ips_state(int tc) {
   //对所有ip更新
   for (int i = 1; i <= n; i++) {
       //处于待分配和占用状态的 IP 地址拥有一个大于零的过期时刻。如果ip的过期时刻大于0，并且过期了
       //处于未分配和过期状态的 IP 地址过期时刻为零，即没有过期时刻。
       if (ip[i].t && ip[i].t <= tc) {
           //如果待分配，则状态变成未分配，且占用者清空，过期时刻清零
           if (ip[i].state == 1) {
               ip[i].state = 0;
               ip[i].owner = "";
               ip[i].t = 0;
           }
           //否则该地址的状态会由占用自动变为过期，且过期时刻清零。
           else {
               ip[i].state = 3;
               ip[i].t = 0;
           }
       }
   }
}

//选取特定状态的ip
int get_ip_by_state(int state) {
   for (int i = 1; i <= n; i++)
       if (ip[i].state == state)
           return i;
   return 0; //否则返回分配失败
}

//通过client找一下有没有用过的ip
int get_ip_by_owner(string client) {
   for (int i = 1; i <= n; i++)
       if (ip[i].owner == client)
           return i;
   return 0;
}

int main() {
   cin >> n >> t_def >> t_max >> t_min >> h;        // 默认过期时间，过期时间的上限，下限，主机名称
   cin >> m;

   while (m--) {
       int tc;//收到报文的时间
       string client, server, type;
       int id, te;//<发送主机> <接收主机> <报文类型> <IP 地址> <过期时刻>
       cin >> tc >> client >> server >> type >> id >> te;
       //按照题目处理细节依次处理
       //判断接收主机是否为本机，或者为 *，若不是，则判断类型是否为 Request，若不是，则不处理；
       if (server != h && server != "*") {     // 接受者不是自己也不是所有人，不处理
           if (type != "REQ") continue;
       }
       // 若类型不是 Discover、Request 之一，则不处理；
       if (type != "DIS" && type != "REQ") continue;
       //若接收主机为 *，但类型不是 Discover，或接收主机是本机，但类型是 Discover，则不处理。
       if ((server == "*" && type != "DIS") || (server == h && type == "DIS")) continue;

       //然后更新一下ip地址池的状态，每次处理一个新的报文前都要更新状态
       update_ips_state(tc);//传入当前时刻

       //处理dis报文
       if (type == "DIS") {
           //检查是否有占用者为发送主机的 IP 地址：
           //若有，则选取该 IP 地址；
           int k = get_ip_by_owner(client);
           //若没有，则选取最小的状态为未分配的 IP 地址；
           if (!k) k = get_ip_by_state(0);//0是未分配
           // 若没有，则选取最小的状态为过期的 IP 地址；
           if (!k) k = get_ip_by_state(3);//过期为3
           // 若没有，则不处理该报文，处理结束
           if (!k) continue;
           //当分配到ip地址以后
           // 将该 IP 地址状态设置为待分配，占用者设置为发送主机；
           ip[k].state = 1, ip[k].owner = client;

           // 若报文中过期时刻为 0 ，则设置过期时刻为 t+Tdef；
           if (!te) ip[k].t = tc + t_def;
           // 否则根据报文中的过期时刻和收到报文的时刻计算过期时间，
           // 判断是否超过上下限：若没有超过，则设置过期时刻为报文中
           // 的过期时刻；否则则根据超限情况设置为允许的最早或最晚的过期时刻；
           else {
               int t = te - tc;//想使用的时间长度
               t = max(t, t_min), t = min(t, t_max);       //！！！！！！！！！！！！！！
               //t既不能小于最低时间，也不能超过最长时间
               ip[k].t = tc + t;
           }
           //应该发送报文了
           cout << h << " " << client << ' ' << "OFR" << ' ' << k << ' ' << ip[k].t << endl;
       }
       //处理request请求
       else {
           //如果接受者不是本机
           if (server != h) {
               // 找到占用者为发送主机的所有 IP 地址，对于其中状态为待分配的，将其状态设置为未分配，并清空其占用者，清零其过期时刻，处理结束；
               for (int i = 1; i <= n; i++) {
                   if (ip[i].owner == client && ip[i].state == 1) {
                       ip[i].state = 0;
                       ip[i].owner = "";
                       ip[i].t = 0;
                   }
               }
               continue;
           }
           // 检查报文中的 IP 地址是否在地址池内，若不是，则向发送主机发送 Nak 报文，处理结束；
           if (!(id >= 1 && id <= n && ip[id].owner == client))
               cout << h << ' ' << client << ' ' << "NAK" << " " << id << ' ' << 0 << endl;
           //如果在地址池内
           else {
               // 无论该 IP 地址的状态为何，将该 IP 地址的状态设置为占用；
               ip[id].state = 2;
               // 与 Discover 报文相同的方法，设置 IP 地址的过期时刻；
               if (!te) ip[id].t = tc + t_def;
               else {
                   int t = te - tc;//想使用的时间长度
                   t = max(t, t_min), t = min(t, t_max);
                   //t既不能小于最低时间，也不能超过最长时间
                   ip[id].t = tc + t;
               }
               //应该发送报文了
               cout << h << ' ' << client << ' ' << "ACK" << ' ' << id << ' ' << ip[id].t << endl;
           }

       }
   }

   return 0;
}

// 202104-4 校门外的树 
// 题意：坐标轴上有n个障碍物，需要在坐标轴上按照一定的规则树，要求，以任意两个障碍物为区间，区间内由树等分，且种树的位置不能有障碍物，答案所且的即是，以坐标轴上最小的障碍物和坐标最大的障碍物构成的区间上种树，有多少种种法
// 思路：动态规划。首先预处理处理每个距离（选取的区间长度）的约数，具体的状态转移看代码

#include <bits/stdc++.h>
using namespace std;

typedef long long LL;       // ！！！！！！！！！！！！ 注意要用LL
const int N = 1010, M = 100010, MOD = 1e9 + 7;
int n;
int a[N], f[N];     // f[i],前i+1个障碍物构成的区间种树的方案数
vector<int> q[M];       // q[i]存q的约数
bool st[M];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);

    //预处理约数
    for (int i = 1; i < M; ++i)
        for (int j = i * 2; j < M; j += i)
            q[j].push_back(i);

    cin >> n;
    for (int i = 0; i < n; ++i) cin >> a[i];

    f[0] = 1;       // init
    for (int i = 1; i < n; ++i) {
        memset(st, 0, sizeof st);       // 哪些约数不能用
        for (int j = i - 1; j >= 0; --j) {      // f[i]被分为f[0~i-1]共i个子区间，滚动数组从右向左
            int d = a[i] - a[j], cnt = 0;       // d为a[i]与d[j]构成区间的长度
            for (int k : q[d])      // 枚举该长度可以设置的等分长度
                if (!st[k]) {
                    ++cnt;
                    st[k] = true;
                }
            st[d] = true;       // ！！！,若之后用公差d,则下一次[i-d,i]就是本此区间，就撞障碍物了
            f[i] = (f[i] + (LL)f[j] * cnt) % MOD;
        }
    }
    cout << f[n - 1];
    return 0;
}




// 	202012-1
#include <bits/stdc++.h>
using namespace std;

int main()
{
	int n;
	cin>>n;
	int y=0;
	
	for(int j=1;j<=n;++j){
		int w,s;
		cin>>w>>s;
		y+=s*w;
	}
	cout<<max(0,y);
	return 0;
}


// 202012-2


//差分数组
#include <bits/stdc++.h>
using namespace std;

const int N=100010;
int diff[N];
int index[N];
map<int,vector<int>> ys;
int n;


int main()
{
	int n;
	cin>>n;
	diff[-1]=0;
	for(int i=1;i<=n;++i){
		int y,s;
		cin>>y>>s;
		ys[y].push_back(s);

	}

	int m=0;
	for(auto kv:ys){
		index[++m]=kv.first;
	}
	int i=0;
	for(auto kv:ys){
		++i;
		for(int j=0;j<kv.second.size();++j){
			int s=kv.second[j];
			if(s==1){
				diff[1]++;
				diff[i+1]--;
			}else{
				diff[i+1]++;
			}
		}
	}

	int res=-1,mx=INT_MIN,sum=0;
	for(int i=1;i<=n;++i){
		sum+=diff[i];
		if(sum>=mx){
			mx=sum;
			res=i;
		}
	}

	cout<<index[res];
	return 0;
}

//法二
#include <bits/stdc++.h>
using namespace std;
const int N=1e5+10;
pair<int,int> ys[N];
int sum[N];
int n;
unordered_set<int> st;
int main()
{
	cin>>n;
   for(int i=1;i<=n;++i)
       cin>>ys[i].first>>ys[i].second;
	// 注意pair<int,int>的排序规则，第一个相同的情况下，第二个小的排在前面
   sort(ys+1,ys+n+1);
   for(int i=1;i<=n;++i)
       sum[i]=sum[i-1]+ys[i].second;

   int mx=-1,res=-1;
   for(int i=1;i<=n;++i){
		int y=ys[i].first;
		// 同一个安全指数做为阈值得出得结果应该是相同得
		// 处理重复的安全指数，只处理第一个，因为对于相同的安全指数，0的在前，1的在后
		if(st.find(y)!=st.end()) continue;
		st.insert(y);
		int sum_1=sum[n]-sum[i-1];
		int sum_0=i-1-sum[i-1];
		if(sum_0+sum_1>=mx){
			mx=sum_0+sum_1;
			res=y;
		}
	}

	cout<<res<<endl;

   return 0;
}



// 202012-3 带配额的文件系统

#include <bits/stdc++.h>
#define  FILE 0
#define  DIRE 1
using namespace std;
typedef long long LL;       // ！！！！！！！！！！！
const LL INF = LLONG_MAX;
LL LD, LR, file_sz;  // 目录配额和后代配额,文件大小
struct node {
   unordered_map<string, node*> dir;
   int type;
   LL ld, lr, sd, sr;
   node(int t) :ld(0), lr(0), sd(0), sr(0), type(t) {}
};

string op, pa;
vector<string> path;
node* root = new node(DIRE);        // ！！！！！！！！！！！！！！！！！！！！
void prase() {
   path.push_back("/");     // ！！！！！！！！！！！！！！
   stringstream ss(pa);
   string s;
   while (getline(ss, s, '/')) if (!s.empty()) path.push_back(s);
}
//对于该指令，若路径所指的文件已经存在，且也是普通文件的，则替换这个文件；
//若路径所指文件已经存在，但是目录文件的，则该指令不能执行成功。
//当路径中的任何目录不存在时，应当尝试创建这些目录；
//若要创建的目录文件与已有的同一双亲目录下的孩子文件中的普通文件名称重复，则该指令不能执行成功。
//另外，还需要确定在该指令的执行是否会使该文件系统的配额变为不满足，如果会发生这样的情况，则认为该指令不能执行成功，反之则认为该指令能执行成功。
bool add(node* r, int u, int old_size) {        // old_size!!!!!!!!!!!!!!!!
   bool end = u + 1 == path.size();
   if (!end && r->lr &&  r->lr < r->sr + file_sz - old_size) return false;  // 后代配额不足
   bool hc = false;        // 记录本次递归是否创建了新节点，若操作失败需要回溯删除！！！！！！！！！！！！！！！！！！！！
   if (r->dir[path[u]]) {      // 目录或文件存在
       if (end && r->dir[path[u]]->type != FILE) return false;
       if (!end && r->dir[path[u]]->type != DIRE) return false;
   } else if (end) {        // 创建文件
       r->dir[path[u]] = new node(FILE);
       hc = true;
   } else {
       r->dir[path[u]] = new node(DIRE);
       hc = true;
   }

   node* next = r->dir[path[u]];
   if (end) {
       LL modify = file_sz - next->sr;
       if ((r->ld && r->ld < r->sd + modify) || (r->lr && r->lr < r->sr + modify)) {
           if (hc) r->dir[path[u]] = nullptr;      // ！！！！！！！！！失败就恢复原样
           return false;
       }
       next->sr = file_sz;     // 若是文件的话，使用sr存储文件的大小
       r->sd += modify;
       r->sr += modify;
       return true;
   }
   if (add(next, u + 1, old_size)) {
       r->sr += file_sz - old_size;
       return true;
   }
   if (hc)r->dir[path[u]] = nullptr;       // ！！！！！！！失败就恢复原样
   return false;
}
// 若该路径所指的文件不存在，则不进行任何操作
// 若该路径所指的文件是目录，则移除该目录及其所有后代文件。
LL del(node* r, int u) {            // 返回所删除的长度，以便回溯进行sr的修改
   if (r->dir[path[u]] == nullptr) return 0;
   bool end = u + 1 == path.size();
   if (!end && r->dir[path[u]]->type != DIRE) return 0;
   if (end) {
       LL res = r->dir[path[u]]->sr;
       if (r->dir[path[u]]->type == FILE) r->sd -= res;   // 若删的是文件，对sd也有影响!!!!!!!！！！！！
       r->dir[path[u]] = nullptr;
       r->sr -= res;
       return res;
   }
   LL res = del(r->dir[path[u]], u + 1);
   r->sr -= res;
   return res;
}

//若路径所指的文件不存在，或者不是目录文件，则该指令执行不成功。
// 若在该目录上已经设置了配额，则将原配额值替换为指定的配额值。
//若在应用新的配额值后，该文件系统配额变为不满足，那么该指令执行不成功。
bool reset(node* r, int u) {
   if (r->dir[path[u]] == nullptr) return false;       // 文件不存在
   bool end = u + 1 == path.size();
   if (!end && r->dir[path[u]]->type != DIRE) return false;        // 走的是目录文件，却遇到普通文件
   node* next = r->dir[path[u]];
   if (end) {
       if (next->type != DIRE) return false;
       if ((LD && LD < next->sd) || (LR && LR < next->sr)) return false;
       next->ld = LD;
       next->lr = LR;
       return true;
   }
   return reset(next, u + 1);
}
LL get_size(node* r,int u) {
   bool end = u + 1 == path.size();
   if (r->dir[path[u]] == nullptr) return 0;
   if (!end && r->dir[path[u]]->type != DIRE) return 0;
   if (end && r->dir[path[u]]->type != FILE) return 0;
   if (end) return r->dir[path[u]]->sr;
   return get_size(r->dir[path[u]], u + 1);
}
int main() {
   ios::sync_with_stdio(false);
   cin.tie(0), cout.tie(0);
   root->dir["/"] = new node(DIRE);
   int T;
   cin >> T;
   while (T--) {
       cin >> op >> pa;
       path.clear();
       prase();

       if (op == "C") {
           cin >> file_sz;
           if (add(root, 0, get_size(root,0))) cout << "Y\n";
           else cout << "N\n";
       } else if (op == "R") {
           del(root, 0);
           cout << "Y\n";
       } else {
           cin >> LD >> LR;
           if (reset(root, 0)) cout << "Y\n";
           else cout << "N\n";
       }
   }
   return 0;
}


// 202012-4 食材运输 
// 题意：n个酒店，n-1个道路连接，构成一棵树，共有k种食材，每个酒店都需要若干种食材，从m个酒店指定酒店选k个作为食材起运点，一个起运点送一种食材，求所有酒店都获得了所需食材得最小时间
// 思路：先预处理每个酒店作为起送点运输不同食材得最短时间，然后二分答案，使用dp+状态压缩检测能否在二分得时间内完成任务
#include <bits/stdc++.h>
using namespace std;

#define x first
#define y second
typedef pair<int, int> PII;
struct node {
    int v, w;
};
const int N = 110, M = 10, S = 1 << M;
int need[N][M], d[N][M];        // d[i][j]是从i点出发运输完食材j的最短时间
int state[N], f[S];// f[10011],代表用最少用多少个点能将状态为10011的食材全部满足
int n, m, k;
vector<node> he[N];

// dfs返回以u为根节点，进行食材f的运送并返回起点所需要的{总路径长度}{从起点到最远需要食材f的酒店的距离}
PII dfs(int u, int fa, int food) {
    PII res(0, -1);     //-1表示以节点u为根节点的子树中不存在需要食材f的点
    if (need[u][food]) res.y = 0;      // 有需要
    for (auto i : he[u]) {
        if(i.v==fa) continue;   // 防止回路
        PII t = dfs(i.v, u, food);
        if (t.y != -1) {
            res.x += t.x + 2 * i.w;
            res.y = max(res.y, t.y + i.w);
        }
    }
    return res;
}
// time时间内能否送完
bool check(int time) {
    // 全覆盖问题
    // 假设有个表格（n行，k列），表格中的元素的值不是1就是0。
    // 此时给定一个确定的表格d,问能否在表格中选取不超过m（ m < n ）行，使得k列中每一列都至少有一个1。
    // 采用dp+状态压缩方法。f[s]表示的集合为：覆盖情况为s的所有行的选取方法。属性为：所有选取方法中所选行数的最小值。
    memset(state, 0, sizeof state);
    // 状态压缩
    for (int i = 1; i <= n; ++i)
        for (int j = 0; j < k; ++j)
            if (d[i][j] <= time) state[i] |= 1 << j;        // 从i出发送完所有食材j可以再time内完成
    
    memset(f, 0x3f, sizeof f);  // !!!!!!!!!!
    // dp
    f[0] = 0;     // 一个起点都不选,全是0,全都不满足
    for (int i = 0; i < 1 << k; ++i)//遍历所有的状态，
        for (int j = 1; j <= n; ++j)        //在原有的状态上添加第i行，产生新状态
            f[i | state[j]] = min(f[i | state[j]], f[i] + 1);
    return f[(1 << k) - 1] <= m;        // 选取的酒店是否超过了m家
}
int main() {
    cin >> n >> m >> k;
    for (int i = 1; i <= n; ++i)
        for (int j = 0; j < k; ++j)
            cin >> need[i][j];
    for (int i = 0; i < n - 1; ++i) {
        int a, b, c;
        cin >> a >> b >> c;
        he[a].push_back({ b,c });
        he[b].push_back({ a,c });
    }
    for(int i=1;i<=n;++i)
        for (int j = 0; j < k; ++j) {
            PII t = dfs(i, -1, j);
            if (t.y != -1) d[i][j] = t.x - t.y;
        }
    // 二分最小答案
    int l = 0, r = 2e8;     // w最多1e6,最多99条边，每个路再一来一回。最大2e8
    while (l < r) {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;
        else l = mid + 1;
    }
    cout << l << endl;
    return 0;
}


//202009-1

#include <bits/stdc++.h>
using namespace std;

int n,x,y;
vector<pair<int,int>> res;
int main()
{
	cin>>n>>x>>y;
	int x1,y1;
	for(int i=1;i<=n;++i){
		cin>>x1>>y1;
		int d=pow(x-x1,2)+pow(y-y1,2);
		res.emplace_back(d,i);
	}
	sort(res.begin(),res.end());
	for(int i=0;i<3;++i)
		cout<<res[i].second<<endl;
		
	return 0;
}


//202009-2

#include <bits/stdc++.h>
using namespace std;
int n,k,t,x1,y1,x2,y2;
bool vaild(int r,int c){
	return r>=x1 && r<=x2 && c>=y1 && c<=y2;
}
int main()
{

	cin>>n>>k>>t>>x1>>y1>>x2>>y2;

	int pass_num=0,stay_num=0;
	for(int i=1;i<=n;++i){
		int last_mx=0,last=0,pass=false;
		for(int j=1;j<=t;++j){
			int r,c;
			cin>>r>>c;
			if(vaild(r,c)){
				++last;
				last_mx=max(last_mx,last);
			}else{
				last=0;
			}
		}
		if(last_mx>0) ++pass_num;
		if(last_mx>=k) ++stay_num;
		

	}
	cout<<pass_num<<endl<<stay_num;

	return 0;
}


// 202009-3
// 思路：结构体存运算类型，输入信号，输出数据，以及需要输入的信号个数，已经输出的信号往那个器件输出
// 注意有多个电路需要处理，注意重置电路信息
// 使用bfs进行电路的模拟，当已经输出的信号个数=需要输出的信号个数，该器件就可以工作，同输出信号传给下个器件

#include <bits/stdc++.h>
using namespace std;

const int N=2510;

struct Logic{
 string type;
 vector<bool> input;        // 输入信号
 vector<int> next;      // 输出连接的器件
 int flag;      // 输入个数
 bool output;
 
 Logic():flag(0){}
 bool work(){
     output=input[0];        // ！！！！！！！
    if(type=="NOT") output=!input[0];
    else if(type=="AND") for(int i=1;i<input.size();++i) output&=input[i];
    else if(type=="OR")for(int i=1;i<input.size();++i)output|=input[i];
    else if(type=="XOR")for(int i=1;i<input.size();++i)output^=input[i];
    else if(type=="NAND"){
        for(int i=1;i<input.size();++i) output&=input[i];
        output=!output;
    }else if(type=="NOR"){
         for(int i=1;i<input.size();++i)output|=input[i];
         output=!output;
     }
     return output;
 }
};

Logic lg[510];      // 器件
vector<int> I[N];       // 输入信号个数
int query[10010][N];        // 询问
int n,m,S;
void build(string s,int i){
     int id=atoi(s.substr(1).c_str());
     if(s[0]=='I'){
         I[id].push_back(i);
     }else{
         lg[id].next.push_back(i);
     }
}
bool work(int cnt){
     queue<int> q;
     for(int i=1;i<=m;++i){
         int num=query[cnt][i];
         for(auto c:I[i]){
             lg[c].input.push_back(num);
             if(lg[c].input.size()==lg[c].flag) q.push(c);      // 输入信号准备齐了，入队列
         }
     }
     if(q.empty()) return false;
     
     int rest=n-q.size();    // ！！！！！！！！！！！
     while(q.size()){
         int sz=q.size();
         for(int i=0;i<sz;++i){
             int id=q.front();
             q.pop();
             lg[id].work();
             int num=lg[id].output;
             for(auto c:lg[id].next){
                 lg[c].input.push_back(num);
                 if(lg[c].input.size()==lg[c].flag) q.push(c),--rest;        // !!!!!!!!!!
                 
             }
         }
     }
     return rest==0;     // ！！！！！！
}
int main()
{
     ios::sync_with_stdio(false);
     cin.tie(0),cout.tie(0);
     int Q;
     cin>>Q;
     while(Q--){
         cin>>m>>n;  // 输入数量，器件个数
         
         for(int j=1;j<=m;++j) I[j].clear();         // !!!!!!!!!!!!
         for(int j=1;j<=n;++j) lg[j].next.clear();   // !!!!!!!!!!!! 注意重置电路图
         
         // 构建电路图
         for(int i=1;i<=n;++i){
             string in;
             cin>>lg[i].type>>lg[i].flag;
             for(int k=0;k<lg[i].flag;++k) {
                 cin>>in;
                 build(in,i);
             }
         }
         // 开始询问，S个输入先存起来
         cin>>S;
         for(int i=1;i<=S;++i)
             for(int j=1;j<=m;++j)
                 cin>>query[i][j];

         bool ok=true;;
         for(int i=1;i<=S;++i){
             for(int j=1;j<=n;++j) lg[j].input.clear();      // ！！！！！！！ 注意重置状态
             if(ok) ok=work(i);
             
             int num,id;
               cin>>num;
               for(int j=0;j<num;++j){
                 cin>>id;
                 if(ok)cout<<lg[id].output<<' ';
             }
             if(ok)cout<<'\n';
         }
         if(!ok) cout<<"LOOP"<<'\n';

     }
     return 0;
}


// 202009-4 星际旅行
// 思路:先计算每个点到圆心的距离，同时记录该点到圆切点的距离（勾股定理）
//  在for for 枚举 计算AOB各边长，利用海伦公式计算高，高即为点到切线的记录，看是否大于半径或是否是个钝角三角形
//      否则，用余弦定理，计算角AOB，再利用勾股定理，计算另外两个直角三角形的角度，相减，计算弧长

#include <bits/stdc++.h>
using namespace std;
int n, m;
const int N = 1e2 + 10, M = 2e3 + 10;
double s[N], p[M][N], ans[M];
double R, d[M], rd[M];

double square(double x) //开平方
{
    return x * x;
}
int main() {
    cin >> n >> m >> R;
    for (int i = 0; i < n; i++) cin >> s[i]; //输入n维黑洞的坐标

    for (int i = 0; i < m; i++) {
        double dis = 0;
        for (int j = 0; j < n; j++) {
            cin >> p[i][j];       //输入每个点的空间坐标
            dis += square(s[j] - p[i][j]);  //计算每个点到黑洞中心的距离
        }
        d[i] = sqrt(dis), rd[i] = sqrt(dis - R * R); //d[i]为点到黑洞中心的距离，rd[i]为当前点与黑洞相切的切线长度
    }

    for (int i = 0; i < m; i++)    //枚举所有点之间的连线关系
    {
        for (int j = i + 1; j < m; j++) {       // 从i+1开始就好，双向的
            double dis = 0;
            for (int k = 0; k < n; k++)    //计算点i和点j之间的距离
            {
                dis += square(p[i][k] - p[j][k]);
            }
            double a = d[i], b = d[j], c = sqrt(dis);   //计算△AOB三条边
            double p = (a + b + c) / 2;     // !!!!!!!!!!!!!!!!!!
            double s = sqrt(p * (p - a) * (p - b) * (p - c));     //海伦公式求面积
            double h = s * 2 / c;           // !!!!!!!!!!!!!!!!!!!
            // AB不过圆   ||  两种钝角三角形的情况，AB连线不经过圆
            if (h >= R || square(a) + dis <= square(b) || square(b) + dis <= square(a))    //情况1
            {
                ans[i] += c, ans[j] += c;    //直接加上A、B间距离c
                continue;
            }
            //情况1不符合要求就进行情况2，需要贴圆走
            double angle1 = acos((square(a) + square(b) - dis) / (2 * a * b));      // 余弦定理
            double angle2 = acos(R / a);        // ！！！！！！！！！！！！！！！！！
            double angle3 = acos(R / b);
            double arc = (angle1 - angle2 - angle3) * R;  //求出弧长
            double len = arc + rd[i] + rd[j];   //求出A,B绕黑洞最短长度
            ans[i] += len, ans[j] += len;        //i,j各加上长度len
        }
    }

    for (int i = 0; i < m; i++)
        printf("%.14lf\n", ans[i]);     // ！！！！！！！！！！！！！！
    return 0;
}



// 202006-1
#include <bits/stdc++.h>
using namespace std;
const int N=1010;
int x[N],y[N];
char type[N];
int main()
{
   int n,m;
   cin>>n>>m;
   for(int i=1;i<=n;++i) cin>>x[i]>>y[i]>>type[i];
   while(m--){
       long long a,b,c;        // !!!!!!!!!!!!!!!!!!!!!!   注意是long long
       cin>>a>>b>>c;
       char uptype;
       if(1.0*(a+b*x[1])/(-c)<y[1]){       // 点在线上方
           uptype=type[1]=='B'?'B':'A';
       }else{              // !!!!!!!!!!!!!!!!!!!!
           uptype=type[1]=='B'?'A':'B';
       }
       int i=2;
       for(;i<=n;++i){
           if(1.0*(a+b*x[i])/(-c)>y[i] && type[i]==uptype) break;
           if(1.0*(a+b*x[i])/(-c)<y[i] && type[i]!=uptype) break;
       }
       if(i<=n) cout<<"No"<<endl;
       else cout<<"Yes"<<endl;
   }
   return 0;
}
// 202006-2


#include <bits/stdc++.h>
using namespace std;
const int N=1e5+10;
unordered_map<int,int> A,B;
int n,a,b;

int main()
{
	cin>>n>>a>>b;

	for(int i=1;i<=a;++i){
		int k,v;
		cin>>k>>v;
		A[k]=v;
	}
	
	for(int i=1;i<=b;++i){
		int k,v;
		cin>>k>>v;
		B[k]=v;
	}
	
	long long res=0;
	for(auto kv:A){
		int k=kv.first;
		int v=kv.second;
		res+=v*B[k];
	}
	cout<<res<<endl;
	return 0;
}

// 202009-3
// 思路：结构体存运算类型，输入信号，输出数据，以及需要输入的信号个数，已经输出的信号往那个器件输出
// 注意有多个电路需要处理，注意重置电路信息
// 使用bfs进行电路的模拟，当已经输出的信号个数=需要输出的信号个数，该器件就可以工作，同输出信号传给下个器件

#include <bits/stdc++.h>
using namespace std;

const int N=2510;

struct Logic{
 string type;
 vector<bool> input;        // 输入信号
 vector<int> next;      // 输出连接的器件
 int flag;      // 输入个数
 bool output;
 
 Logic():flag(0){}
 bool work(){
     output=input[0];        // ！！！！！！！
    if(type=="NOT") output=!input[0];
    else if(type=="AND") for(int i=1;i<input.size();++i) output&=input[i];
    else if(type=="OR")for(int i=1;i<input.size();++i)output|=input[i];
    else if(type=="XOR")for(int i=1;i<input.size();++i)output^=input[i];
    else if(type=="NAND"){
        for(int i=1;i<input.size();++i) output&=input[i];
        output=!output;
    }else if(type=="NOR"){
         for(int i=1;i<input.size();++i)output|=input[i];
         output=!output;
     }
     return output;
 }
};

Logic lg[510];      // 器件
vector<int> I[N];       // 输入信号个数
int query[10010][N];        // 询问
int n,m,S;
void build(string s,int i){
     int id=atoi(s.substr(1).c_str());
     if(s[0]=='I'){
         I[id].push_back(i);
     }else{
         lg[id].next.push_back(i);
     }
}
bool work(int cnt){
     queue<int> q;
     for(int i=1;i<=m;++i){
         int num=query[cnt][i];
         for(auto c:I[i]){
             lg[c].input.push_back(num);
             if(lg[c].input.size()==lg[c].flag) q.push(c);      // 输入信号准备齐了，入队列
         }
     }
     if(q.empty()) return false;
     
     int rest=n-q.size();    // ！！！！！！！！！！！
     while(q.size()){
         int sz=q.size();
         for(int i=0;i<sz;++i){
             int id=q.front();
             q.pop();
             lg[id].work();
             int num=lg[id].output;
             for(auto c:lg[id].next){
                 lg[c].input.push_back(num);
                 if(lg[c].input.size()==lg[c].flag) q.push(c),--rest;        // !!!!!!!!!!
                 
             }
         }
     }
     return rest==0;     // ！！！！！！
}
int main()
{
     ios::sync_with_stdio(false);
     cin.tie(0),cout.tie(0);
     int Q;
     cin>>Q;
     while(Q--){
         cin>>m>>n;  // 输入数量，器件个数
         
         for(int j=1;j<=m;++j) I[j].clear();         // !!!!!!!!!!!!
         for(int j=1;j<=n;++j) lg[j].next.clear();   // !!!!!!!!!!!! 注意重置电路图
         
         // 构建电路图
         for(int i=1;i<=n;++i){
             string in;
             cin>>lg[i].type>>lg[i].flag;
             for(int k=0;k<lg[i].flag;++k) {
                 cin>>in;
                 build(in,i);
             }
         }
         // 开始询问，S个输入先存起来
         cin>>S;
         for(int i=1;i<=S;++i)
             for(int j=1;j<=m;++j)
                 cin>>query[i][j];

         bool ok=true;;
         for(int i=1;i<=S;++i){
             for(int j=1;j<=n;++j) lg[j].input.clear();      // ！！！！！！！ 注意重置状态
             if(ok) ok=work(i);
             
             int num,id;
               cin>>num;
               for(int j=0;j<num;++j){
                 cin>>id;
                 if(ok)cout<<lg[id].output<<' ';
             }
             if(ok)cout<<'\n';
         }
         if(!ok) cout<<"LOOP"<<'\n';

     }
     return 0;
} 

// 202006-4 1246

#include <bits/stdc++.h>
using namespace std;
typedef long long LL;       // ！！！！！！！！！！！！！！！！！
const int N = 14, MOD = 998244353;

int n;
string S;
int id[100];        // id[x]将x映射到0~13
vector<int> vers{
    1, 2, 4, 6, 16, 26, 41, 42, 44,
    46, 61, 62, 64, 66
};
// 上面的数字进行运算后可以得出下面的贡献，两位数字的运算后的数字只有一个
// 如16->26（不考虑2，6，4，64的原因是，16可以拆分为1和6，而1->2 , 6->64, 6, 4)
// 故两位数字运算后只转移到前一位数字幂的后一位和后一位数字幂的前一位

// 当s的长度大于2时，我们可以根据转移的规律，逆推到s只有两位的情况。
// s在整个序列的情况，可能分为s前面是1，s前面是6，或者s前面不是1或6.
// 因为4->16 6->64,都会产生两位，所以s的首个数字可能是16中的6，以及64中的6，由于s只是其中的一部分，所以我们要补上1和6
vector<vector<int>> g{
    {2}, {4}, {1, 6, 16}, {6, 4, 64}, {26},
    {46}, {62}, {64}, {61}, {66}, {42},
    {44}, {41}, {46}
};
int tr[N][N];       // tr的第j列是对ver[j]的贡献

void init() {
    memset(id, -1, sizeof id);      // ！！！！！！！！！！！！
    for (int i = 0; i < N; i++) id[vers[i]] = i;        // 映射
    //求转移矩阵
    for (int i = 0; i < N; i++)
        for (auto x : g[i])
            tr[i][id[x]] ++;
}

void mul(int c[][N], int a[][N], int b[][N]) {
    static int tmp[N][N];
    memset(tmp, 0, sizeof tmp);
    for (int i = 0; i < N; i++)     // 行
        for (int j = 0; j < N; j++)     // 列
            for (int k = 0; k < N; k++)     // 行×列
                tmp[i][j] = (tmp[i][j] + (LL)a[i][k] * b[k][j]) % MOD;
    memcpy(c, tmp, sizeof tmp);
}

int qmi(int k, int id) {
    if (id == -1) return 0;
    int res[N][N] = { 0 }, w[N][N];
    memcpy(w, tr, sizeof w);    //！！！！！！！！！！！！！！
    res[0][0] = 1;      // 初始，使用res的首行充当0时刻的行向量

    while (k) {
        if (k & 1) mul(res, res, w); //res=res*w
        mul(w, w, w);   // w=w*w;
        k >>= 1;
    }
    return res[0][id];
}

string get(string str) {
    string res;
    for (int i = 0; i < str.size(); i++)
        if (str[i] == '2') res += '1';      // 2由1
        else if (str[i] == '4') res += '2'; // 4由2
        else if (str[i] == '1') {       // 1只能有4来 -》16
            if (i + 1 == str.size() || str[i + 1] == '6') res += '4', i++;
            else return "";
        } else {        // 6 6只能由64来
            if (i + 1 == str.size() || str[i + 1] == '4') res += '6', i++;
            else return "";
        }
    return res;
}

int dfs(int k, string& str) {
    if (str.size() <= 2) return qmi(k, id[stoi(str)]);  //!!!!!!!!!!!!!!!!!!!!!!!
    int res = 0;
    for (string s : {"", "1", "6"}) {       // 前补1或6
        auto t = get(s + str);
        if (t.size()) res = (res + dfs(k - 1, t)) % MOD;        // ！！！！！！！
    }
    return res;
}

int main() {
    init();
    cin >> n >> S;
    cout << dfs(n, S) << endl;
    return 0;
}


//201912-1
#include <bits/stdc++.h>
using namespace std;
int res[4];
bool vaild(int k){
	if(k%7==0) return false;
	while(k){
		if(k%10==7) return false;
		k/=10;
	}
	return true;
}
int main()
{
	int n,index=0;
	cin>>n;
	int cnt=0;
   for(int i=1;cnt<n;++i){
	
		if(!vaild(i)){
			++res[(i-1)%4];
		}else{
			cnt++;
		}
	}
	cout<<res[0]<<endl;
	cout<<res[1]<<endl;
	cout<<res[2]<<endl;
	cout<<res[3]<<endl;
	return 0;
}

// 201912-2

#include <bits/stdc++.h>
using namespace std;

typedef pair<int,int> PII;
set<PII> gab;
const int N=1010;

int main()
{
   int n;
   cin>>n;
   while(n--){
       int x,y;
       cin>>x>>y;
       gab.insert({x,y});
   }
   int ans[5]={0};
   for(auto k:gab){
       int x=k.first,y=k.second;
       if(gab.find({x+1,y})==gab.end()) continue;
       if(gab.find({x-1,y})==gab.end()) continue;
       if(gab.find({x,y+1})==gab.end()) continue;
       if(gab.find({x,y-1})==gab.end()) continue;
       int cnt=0;
       if(gab.find({x+1,y+1})!=gab.end()) ++cnt;
       if(gab.find({x-1,y+1})!=gab.end()) ++cnt;
       if(gab.find({x-1,y-1})!=gab.end()) ++cnt;
       if(gab.find({x+1,y-1})!=gab.end()) ++cnt;
       ans[cnt]++;
   }
   for(int i=0;i<5;++i) cout<<ans[i]<<endl;
   return 0;
}

// 201912-3 化学方程式

#include <bits/stdc++.h>
using namespace std;
struct node{
 string ele;
 int cnt;
 node(string e,int c=1):ele(e),cnt(c){}

};
unordered_map<string,int> cnt1,cnt2;
int CulNum(string& s,int& i){
 int res=0;
 while(i<s.size() && isdigit(s[i])){
     res=res*10+s[i]-'0';
     ++i;
 }
 --i;
 return max(res,1);
}
void work(string s,unordered_map<string,int>& mp){
 stringstream ss(s);
 while(getline(ss,s,'+')){      // 分割
     int i=0,k1=0;
     int k=CulNum(s,i);      // k是系数，k1是角标

     stack<node> st;
     string elem="null";

     bool over=false;         // 上一个node是否是(或)
     for(++i;i<s.size();++i){
         if(s[i]=='('){
             if(!over) st.push({elem,k1});
             st.push({"("});
             over=true;
         }else if(s[i]==')'){
             if(!over)st.push({elem,k1});         //  ！！！！！！！！！！！！
             ++i;         //  ！！！！！！！！！！！！
             k1=CulNum(s,i);
             // 结算括号内的，同时乘以角标
             vector<node> tmp;
             while(st.top().ele!="("){
                 st.top().cnt*=k1;
                 tmp.push_back(st.top());
                 st.pop();
             }
             st.pop();       // 弹出(
             for(auto t:tmp) st.push(t);
             over=true;
         }else if(isupper(s[i])){
             // 遇到元素开始结算上一个node，因为（）也会结算，所以上一个是（）的话就不结算了
             if(!over) st.push({elem,k1});
             k1=1;       // 这里记得置为1,角标为1不显示

             if(i<s.size() &&islower(s[i+1])){      // Ba
                 elem=string(1,s[i])+s[i+1];
                  i++;
             }else         // C
                 elem=string(1,s[i]);
             over=false;
         }else if(isdigit(s[i])){
             k1=CulNum(s,i);
         }
     }
         // 栈中保存了所有结算完的节点，栈底的节点是无效节点
     if(!over)st.push({elem,k1});        // ！！！！！！！！！！！！！！！！
     while(st.size()>1){
         auto t=st.top();st.pop();
         mp[t.ele]+=t.cnt*k;
     }
 }
}

int main()
{
 int n;
 cin>>n;
 while(n--){
     cnt1.clear(),cnt2.clear();
     string s;
     cin>>s;
     stringstream ss(s);

     getline(ss,s,'=');
     work(s,cnt1);
     getline(ss,s,'=');
     work(s,cnt2);

     cout<<((cnt1==cnt2)?'Y':'N')<<endl;
 }

 return 0;
}


//201912-4 区块链
// 题意：n个节点，m条边，每个节点都有一个链，初始链上都有个编号为0的块，当某个节点的链更新时，会将其链发送给相邻节点，若相邻节点，判断：收到的链比自己更长，将其作为自己的链，若一样长，但收到的链的最后一个块的编号更小，则作为自己的链。同时节点之间链的传播时间都为t。最后输出每个节点的链。
// 思路：由于传播时长相等，故使用queue,queue与时间同步更新，先更新再处理
// 对于链的不同时刻的修改都要保存，因为队列中尚有节点要还没开的及对之间的节点状态进行更新
#include <bits/stdc++.h>
using namespace std;
struct Op {
    int x, y, t, id;    // x发给y的链id,将在t时刻到达
};
const int N = 510;
vector<int> he[N];
vector<vector<int>> link(1,vector<int>(1,0));   // 二维数组，存储初始链条
int dat[N];     // 存储节点x最新的链的下标
int n, m, t, k;
queue<Op> q;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);

    cin >> n >> m;
    for (int i = 1; i <= n; ++i) dat[i] = 0;    // 所有链初始的状态都一眼，为link[0]
    for (int i = 1; i <= m; ++i) {
        int a, b; cin >> a >> b;
        if (a != b) he[a].push_back(b), he[b].push_back(a);
    }

    cin >> t >> k;
    string s;
    getline(cin, s);        // !!!!!!!!!!!!!!，cin优化后不能用getchar了
    while (k--) {
        int arr[3];
        getline(cin, s);
        stringstream ss(s);
        int cnt = 0;
        while (ss >> arr[cnt])++cnt;
        int a = arr[0], b = arr[1], c = arr[2];     // a节点在b时刻产生了节点c
       
        // 处理b时刻及之前的操作
        while (!q.empty() && q.front().t <= b) {
            Op nd = q.front(); q.pop();
            int idx = nd.id, idy = dat[nd.y];       // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!,此时的idx应为nd.id
            if(link[idy].size()>link[idx].size() ||
                link[idy].size()==link[idx].size() && link[idy].back()<=link[idx].back())
                continue;
            else
                dat[nd.y] = nd.id;

            for (int i : he[nd.y]) {
                if (nd.x == i) continue;
                q.push( { nd.y, i, nd.t + t, nd.id });
            }

        }
        bool update = false;        // !!!!!!!!!!!!!!!!
        if (cnt == 3) {
            // a节点b时刻产生了节点c
            link.push_back(link[dat[a]]);       // 产生新的主链
            link.back().push_back(c);
            dat[a] = link.size()-1;
            update = true;
        } else {
            // 查询a节点b时刻处理完后的主链
            cout << link[dat[a]].size();
            for (int i : link[dat[a]]) cout << ' ' << i;
            cout << '\n';
        }
        if (update)     // 更新了
            for (int i : he[a])
                q.push({ a,i,b + t,dat[a] });

    }
    return 0;
}


//201909-1

#include <bits/stdc++.h>
using namespace std;


int a[1010];
pair<int,int> cnt[1010];
int main()
{
	int n,m;
	cin>>n>>m;

	for(int i=1;i<=n;++i){
		cin>>a[i];
		cnt[i].second=i;
		for(int j=1;j<=m;++j){
			int t;
			cin>>t;
			a[i]+=t;
			cnt[i].first-=t;
		}
	}
	
	int sum=0;
	for(int i=1;i<=n;++i) sum+=a[i];
	
	sort(cnt+1,cnt+1+n,[](const pair<int,int>& x,const pair<int,int>& y){
		if(x.first!=y.first) return x.first<y.first;
		return x.second>y.second;
	});
	cout<<sum<<' '<<cnt[n].second<<' '<<cnt[n].first;

	return 0;
}


//201909-2

#include <bits/stdc++.h>
using namespace std;
int n;
bool fall[1010];
int main()
{
	cin>>n;
	int sum=0;
	for(int i=0;i<n;++i){
		int m,a0;
		cin>>m>>a0;
		bool is_fall=false;
		for(int j=2;j<=m;++j){
			int t;
			cin>>t;
			if(t<=0){
				a0+=t;
			}else if(a0>t){
				is_fall=true;
				a0=t;
			}
		}
		fall[i]=is_fall;
		sum+=a0;
	}
	int cnt1=0,cnt2=0;
	for(int i=0;i<n;++i){
		if(fall[i]) ++cnt1;
		if(fall[i] && fall[(i+1)%n] && fall[(i+2)%n] ) ++cnt2;
	}
	cout<<sum<<' '<<cnt1<<' '<<cnt2<<endl;
	return 0;
}


// 201909-3
// 题意：将一个图片压缩，p*q个像素平均压缩为一个，题目中只用到背景色，没有前景色，每一个像素块后必须紧跟一个格式化的空格
// 你需要将输出中的所有字符转换为 ASCII 编码转义后的格式再进行输出。注意数字字符的实际ASCII码

#include <bits/stdc++.h>
using namespace std;
unsigned char pic[1080][1920][3];
int m,n,p,q,sz;

// 将字符形式的16进制转为数字
unsigned char GetPixel(char a,char b){
 return (unsigned char)((isalpha(a)?(10+a-'a'):(a-'0'))*16+(isalpha(b)?(10+b-'a'):(b-'0')));
}
void OutStr(const string& str){
 for(auto c:str) cout<<"\\x"<<hex<<uppercase<<setw(2)<<int(c);      //！！！！！！！！！！！！！！！！
}
// 获取像素块的平均RGB
void GetRGB(int row,int col,int& r,int& g,int &b){
 for(int i=row;i<row+p;++i){
     for(int j=col;j<col+q;++j){
         r+=pic[i][j][0];
         g+=pic[i][j][1];
         b+=pic[i][j][2];
     }
 }
 r/=sz,g/=sz,b/=sz;
}
int main()
{

 cin>>m>>n>>q>>p;
 sz=p*q;
 cout.fill('0');     // 格式输出的空白符号设置为0!!!!!!!!!!!!!!!!!
 
 string s;
 for(int i=0;i<n;++i)
     for(int j=0;j<m;++j){
         cin>>s;
         if(s.size()==2){    // #a->#aaaaaa
             s+=string(5,s[1]);
         }else if(s.size()==4){  // #abc ->#aabbcc
             s='#'+string(2,s[1])+string(2,s[2])+string(2,s[3]);
         }
         for(int k=0;k<3;++k) pic[i][j][k]=GetPixel(s[1+k+k],s[2+k+k]);
     }
 int R=0,G=0,B=0,r=0,g=0,b=0;
 for(int i=0;i<n;i+=p){
     for(int j=0;j<m;j+=q){
         R=0,G=0,B=0;
         GetRGB(i,j,R,G,B);
         if(!(R==r && G==g && B==b)){    // 与同行的上段颜色不同
             if(R==0&&G==0&&B==0) OutStr(string(1,char(27))+"[0m");  // 重置默认色！！！！！！！！！！！
             else {
                 OutStr(string(1,char(27))+"[48;2;");
                 OutStr(to_string(R)),OutStr(";"),OutStr(to_string(G)),OutStr(";"),OutStr(to_string(B)),OutStr("m");
             }
             r=R,g=G,b=B;
         }
         OutStr(" ");       // ！！！！！！！！！！！！
     }
     if(R || G||B) OutStr(string(1,char(27))+"[0m");     // 若该行的最后一个像素块颜色不是默认值则输出 ESC[0m 的格式化表示。
     r=g=b=0;        // !!!!!!!!!!!!
     OutStr("\n");      // !!!!!!!!!!!!!!!!!!!!!
 }
 return 0;
}


// 201909-4 推荐系统
//题意： 0~m-1的m类商品，每类商品都有编号不同的n个商品，初始时各类商品的编号和得分都相同，有如下操作，1.在t类商品中增加编号为id的商品，得分为s；2.在t类商品中删除编号为id的商品，3.各类商品中选出不超过K个，得分最大的商品
//思路：使用堆存储所有的商品，按照题目定义排序规则，同时使用unordered_map<int,unordered_set<int>>标记删除的商品
#include <bits/stdc++.h>
using namespace std;
const int M=50;
struct node{
    int s,type,id;
    bool operator<(const node& a) const{
        if(s!=a.s) return s<a.s;
        if(type==a.type) return id>a.id;    // ！！！！！！！！！
        return type>a.type;
    }
};
priority_queue<node> q;
unordered_map<int,unordered_set<int>> mp;      // 是否被删除
vector<int> res[M];
int limit[M];
int n,m;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin>>m>>n;
    for(int i=0;i<n;++i){
        int id,s;
        cin>>id>>s;
        for(int j=0;j<m;++j){       // ！！！！！
            q.push({s,j,id});
        }
    }
    int opn,op,type,id,s;
    cin>>opn;
    while(opn--){
        cin>>op;
        if(op==1){      // add
            cin>>type>>id>>s;
            q.push({s,type,id});
            if(mp[type].find(id)!=mp[type].end()) mp[type].erase(id);    
        }else if(op==2){    // del
            cin>>type>>id;
            mp[type].insert(id);    // ！！！！！！！！！
        }else{      // ask
            vector<node> tmp;
            int all,full_cnt=m;
            cin>>all;
            for(int i=0;i<m;++i) cin>>limit[i];
            while(all-- && full_cnt){
                auto t=q.top();
                q.pop();
                if(mp[t.type].find(t.id)!=mp[t.type].end()) {
                    mp[t.type].erase(t.id);     //已出堆，删除标记取消
                    ++all;
                    continue;
                }
                tmp.push_back(t);
                if(limit[t.type]==0){       // ！！！！！！！！！！！！
                    ++all;
                    continue;
                }
                if(--limit[t.type]==0) --full_cnt;
                res[t.type].push_back(t.id);
            }
            
            for(auto t:tmp) q.push(t);
            for(int i=0;i<m;++i){
                if(res[i].empty()) cout<<-1;
                for(int j=0;j<res[i].size();++j) cout<<res[i][j]<<' ';
                cout<<'\n';
                res[i].clear();
            }
        }
    }
    return 0;
}




//201903-1


#include <bits/stdc++.h>
using namespace std;

const int N=1e5+10;
int a[N];
int n;

int main()
{
	cin>>n;
	for(int i=1;i<=n;++i) cin>>a[i];
	sort(a+1,a+1+n);
	printf("%d ",a[n]);
	double avge=0;
	if(n%2){
		avge=a[(n+1)/2];
	}else{
		avge=(a[n/2]+a[(n/2)+1])/2.0;
	}
	if(avge==(int)avge) printf("%d ",(int)avge);
	else printf("%.1f ",avge);
	printf("%d\n",a[1]);
	return 0;
}


//201903-2

#include <bits/stdc++.h>
using namespace std;

int main()
{
   int n;
   cin>>n;
   deque<int> nums;        // !!!!! deque
   deque<char> ops;
   while(n--){
       char num,op;        // ！！！！！ char类型
       cin>>num;
       nums.push_back(num-'0');        // !!!!!!!!! 注意-‘0’
       for(int i=0;i<3;++i){
           cin>>op>>num;
           if(op=='x'){
               int t=nums.back();nums.pop_back();
               nums.push_back(t*(num-'0'));
           }else if(op=='/'){
               int t=nums.back();nums.pop_back();
               nums.push_back(t/(num-'0'));
           }else{
               nums.push_back(num-'0');        // 注意push!!!!!!!!!!!!!!
               ops.push_back(op);
           }
       }
       while(!ops.empty()){
           int a=nums.front();nums.pop_front();
           int b=nums.front();nums.pop_front();
           char op=ops.front();ops.pop_front();
           if(op=='+') nums.push_front(a+b);
           else nums.push_front(a-b);
       }
       if(nums.front()==24) cout<<"Yes"<<endl;
       else cout<<"No"<<endl;
       nums.pop_front();
   }

   return 0;
}



// 201903-3 损坏的RAID5


#include <bits/stdc++.h>
#include <bitset>
using namespace std;
const int N = 1010;
string raid[N];
int n, s, l, len;
bool ok = false;
string to_bits(char c) {
   int k;
   if (isdigit(c)) k = c - '0';
   else k = 10 + c - 'A';
   string res;
   while (k) {
       res.push_back(k % 2 + '0');
       k /= 2;
   }
   for (int i = res.size(); i < 4; ++i) res += '0';
   reverse(res.begin(), res.end());
   return res;
}
string to_string(bitset<32>& b) {
   string res;
   for (int i = 31; i >= 0; i -= 4) {
       int k = 0;
       for (int j = 0; j < 4; ++j)
           k = (k + b[i - j]) << 1;
       k >>= 1;
       if (k < 10) res += k + '0';
       else res += 'A' + k - 10;
   }
   return res;
}
string recover(int c) {
   int k = 0;
   bitset<32> res;
   for (int i = 0; i < n; ++i)
       if (raid[i].empty())
           k = i;
       else {
           string s;
           for (int j = 0; j < 8; ++j) s += to_bits(raid[i][c + j]);
           res ^= bitset<32>(s);
       }
   return to_string(res);
}
void work(int b) {
   int k = b;
   b = b / s;      // 在b号条带上      // !!!!!!!!!!!
   int t = b % (n - 1);
   b /= (n - 1);     // 在第b行上      // !!!!!!!!!!!!

   int st = (n - 1) - (b % n);     // 该行的校验块所在的磁盘块
   for (int i = 0; i <= t; ++i) st = (st + 1) % n; // 走到存储数据的条带上
   int p = b * s * 8 + (k % s) * 8;       // 所在块的首位数据的地址

   if (raid[st].empty()) {
       if (!ok) cout << '-' << endl;
        else cout << recover(p) << endl;
       return;
   }
   for (int i = 0; i < 8; ++i) cout << raid[st][p + i];
   cout << endl;

}


int main() {
   ios::sync_with_stdio(false);
   cin.tie(0), cout.tie(0);
   cin >> n >> s >> l;
   if (l == n - 1)  ok = true;

   for (int i = 1; i <= l; ++i) {
       int p;
       cin >> p;
       cin >> raid[p];
       len = raid[p].size();
   }

   int m;
   cin >> m;
   while (m--) {
       int b; cin >> b;
       if (b >= (n - 1) * len / 8) cout << '-' << endl;
       else  work(b);
   }
   return 0;
}


// 201903-4 消息传递接口
// 多路扫描,模拟
#include <bits/stdc++.h>
using namespace std;    
struct Msg{
    char c;
    int id;
};
const int N=1e4+10;
queue<Msg> msg[N];

int n;
bool is_lock(){     // 死锁返回1，否则返回0
    bool vis=false; // 是否匹配过
    bool empty=true;    // 是否全空
    for(int i=0;i<n;++i){
        if(msg[i].empty()) continue;
        empty=false;
        auto t=msg[i].front();
        if(t.c=='R'){
            if(msg[t.id].empty()) return true;      // 对方空了
            if(msg[t.id].front().c=='R' && msg[t.id].front().id==i) return true;    // 双方互等，环
            if(msg[t.id].front().c=='S' && msg[t.id].front().id==i){        // 对上了
                msg[i].pop();
                msg[t.id].pop();
                vis=true;
            }
        }else{// 这里没有互相等
            if(msg[t.id].empty()) return true;  // 对方空了
            if(msg[t.id].front().c=='R' && msg[t.id].front().id==i){
                msg[i].pop();
                msg[t.id].pop();
                vis=true;
            }
        }
    }
    if(vis) return is_lock();
    if(empty) return false;
    return true;
}
int main()
{
    int T;
    cin>>T>>n;
    getchar();  // ！！！！！！
    while(T--){
        for(int i=0;i<n;++i){
            msg[i]=queue<Msg>();        // 清空
            string s;
            getline(cin,s);
            stringstream ss(s);
            char c;int t;
            while(ss>>s){
                sscanf(s.c_str(),"%c%d",&c,&t);     // ！！！！！！！
                msg[i].push({c,t});
            }
        }
        cout<<is_lock()<<endl;
    }
    return 0;
}



// 	201812-1
#include <bits/stdc++.h>
using namespace std;

int r,y,g,n;

int main()
{
	cin>>r>>y>>g;
	cin>>n;
	int sum=0;
	while(n--){
		int k,t;
		cin>>k>>t;
		if(k==0){
			sum+=t;
		}else if(k==1){
			sum+=t;
		}else if(k==2){
			sum+=t+r;
		}
	}
	cout<<sum<<endl;
	return 0;
}


//201812-2


#include <bits/stdc++.h>
using namespace std;

int a[3];
int k,t,n;

void recalu(int time){
	if(k==3) k=2;
	else if(k==2) k=3;
	int i=k-1,rest=t;
	while(time>=0){
		if(rest>time){
			k=i+1;
			if(k==2) k=3;
			else if(k==3) k=2;
			t=rest-time;
			return;
		}
		time-=rest;
		i=(i+1)%3;
		rest=a[i];
	}
}
int main()
{
	// a的顺序为：红 绿 黄
	// 输入顺序为：红 黄 绿
	cin>>a[0]>>a[2]>>a[1];
	int all=a[0]+a[1]+a[2];
	cin>>n;
	long long sum=0;

	while(n--){
		
		cin>>k>>t;
		if(k) recalu(sum%all);
	
		if(k==0){
			sum+=t;
		}else if(k==1){ // 红
			sum+=t;
		}else if(k==2){     // 黄
			sum+=t+a[0];
		}   // k==3 绿
	}
	cout<<sum<<endl;
	return 0;
}

//或


#include <bits/stdc++.h>
using namespace std;
int r,y,g,n;

int main()
{
	cin>>r>>y>>g>>n;
	int rg=r+g,rgy=r+g+y;
	long long sum=0;        // !!!!!!!!!!!! LL
	// 红[0,r)  绿[r,r+g)  黄[r+y,r+g+y)
	while(n--){
		int k,t;
		cin>>k>>t;
		if(k==0){
			sum+=t;
			continue;
		}
		// 重新映射到[0,r+g+y)
		if(k==1) t=(r-t+sum)%rgy;   // 红
		else if(k==2) t=(rgy-t+sum)%rgy;    // 黄
		else if(k==3) t=(rg-t+sum)%rgy;     // 绿

		if(t<r){        // 处于红灯时间
			sum+=r-t;
		}else if(t>=rg){     // 处于黄灯时间!!!!!!!!!!!!!! >=
			sum+=rgy-t+r;
		}
	}
	cout<<sum<<endl;
	return 0;
}


// 201812-3 CIDR合并
// 看好题意，不要跳读，理解号匹配集
// 用好sscanf和sprintf，利用.的个数和/是否出现区分不同类型

#include <bits/stdc++.h>
using namespace std;
typedef unsigned int ui;    // !!!!!!!!!!! 记得用ui类型!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
const int N=1e5+10;
string a[N];
int n;
char buf[20];

string standard(string s){
 string res;
 ui x1=0,x2=0,x3=0,x4=0,x5=0;     // !!!!!!!!!!!!!!pos=0
 int dot_cnt=0,pos=0;        // pos是int类型要初始化为0
 bool xie=s.find('/')!=s.npos;

 while((pos=s.find('.',pos))!=s.npos) ++dot_cnt,++pos;

 if(dot_cnt==3 && xie){      //标准型
     sscanf(s.c_str(),"%u.%u.%u.%u/%u",&x1,&x2,&x3,&x4,&x5);
 }else if(dot_cnt<3 && xie){        // 省略后缀型
     if(dot_cnt==2){
         sscanf(s.c_str(),"%u.%u.%u/%u",&x1,&x2,&x3,&x5);
     }else if(dot_cnt==1){
         sscanf(s.c_str(),"%u.%u/%u",&x1,&x2,&x5);
     }else{
         sscanf(s.c_str(),"%u/%u",&x1,&x5);
     }
 }else{      // 省略长度型
     x5=(dot_cnt+1)*8;
     if(dot_cnt==3) sscanf(s.c_str(),"%u.%u.%u.%u",&x1,&x2,&x3,&x4);
     else if(dot_cnt==2) sscanf(s.c_str(),"%u.%u.%u",&x1,&x2,&x3);
     else if(dot_cnt==1) sscanf(s.c_str(),"%u.%u",&x1,&x2);
     else sscanf(s.c_str(),"%u",&x1);
 }
 // 统一补上前缀0
 sprintf(buf,"%03u.%03u.%03u.%03u/%02u",x1,x2,x3,x4,x5);

 return buf;
}
// small 是否为big的字集
bool check(string& big,string& small){
 if(big==small) return true;

 ui x1,x2,x3,x4,x5;
 ui y1,y2,y3,y4,y5;
 sscanf(big.c_str(),"%u.%u.%u.%u/%u",&x1,&x2,&x3,&x4,&x5);
 sscanf(small.c_str(),"%u.%u.%u.%u/%u",&y1,&y2,&y3,&y4,&y5);
 if(x5==y5) return false;        // 同级不可能

 ui xl=x1*pow(256,3)+x2*pow(256,2)+x3*pow(256,1)+x4;     // 左区间，即ip的十进制
 ui yl=y1*pow(256,3)+y2*pow(256,2)+y3*pow(256,1)+y4;
 ui xr=xl|((1<<(32-x5))-1),yr=yl|((1<<(32-y5))-1);
 return xl<=yl && xr>=yr;
}
string merge(string& big,string& small){
 ui x1,x2,x3,x4,x5;
 ui y1,y2,y3,y4,y5;
 sscanf(big.c_str(),"%u.%u.%u.%u/%u",&x1,&x2,&x3,&x4,&x5);
 sscanf(small.c_str(),"%u.%u.%u.%u/%u",&y1,&y2,&y3,&y4,&y5);
 if(x5!=y5) return "";       // 不同级，不用合并
 ui z5=x5-1;

 ui xl=x1*pow(256,3)+x2*pow(256,2)+x3*pow(256,1)+x4;
 ui yl=y1*pow(256,3)+y2*pow(256,2)+y3*pow(256,1)+y4;


 ui xr=xl|((1<<(32-x5))-1),yr=yl|((1<<(32-y5))-1);
 ui zl=xl,zr=xl|((1ll<<(32-z5))-1);

 if(zl==xl && zr==yr && xr+1>=yl){
     sprintf(buf,"%03u.%03u.%03u.%03u/%02u",x1,x2,x3,x4,z5);
     return buf;
 }

 return "";

}
int main()
{
 ios::sync_with_stdio(false);
 cin.tie(0),cout.tie(0);

 cin>>n;
 for(int i=0;i<n;++i){
     string s;
     cin>>s;
     a[i]=standard(s);
 }
 sort(a,a+n);        // ！！！！！！！！

 vector<string> mer;
 mer.push_back(a[0]);        // ！！！！
 for(int i=1;i<n;++i){
     if(check(mer.back(),a[i])) continue;
     mer.push_back(a[i]);
 }
 vector<string> res;
 res.push_back(mer[0]);
 for(int i=1;i<mer.size();++i){
     auto s=merge(res.back(),mer[i]);
     if(s.empty()){      // 不能合并
         res.push_back(mer[i]);
     }else if(res.size()>1){     //  ！！！！！！！！！
         res.pop_back();
         mer[i]=s;
         --i;
     }else{
         res.pop_back();
         res.push_back(s);
     }
 }
 for(int i=0;i<res.size();++i){
     int x1=0,x2=0,x3=0,x4=0,x5=0;
     sscanf(res[i].c_str(),"%d.%d.%d.%d/%d",&x1,&x2,&x3,&x4,&x5);
     sprintf(buf,"%d.%d.%d.%d/%d",x1,x2,x3,x4,x5);
     cout<<buf<<endl;
 }

 return 0;
}




//  201812-4 数据中心 
// 题意：在最小生成树上找出最大的边
#include <bits/stdc++.h>
using namespace std;
const int N=5*1e4+10;

int fa[N];

int n,m,root;
struct node{
    int u,v,w;
    bool operator<(const node& a)const{
        return w>a.w;
    }
};
int find(int x){
    if(x!=fa[x]) fa[x]=find(fa[x]);
    return fa[x];
}
priority_queue<node> q;
int main()
{
    cin>>n>>m>>root;

    for(int i=1;i<=n;++i) fa[i]=i;

    for(int i=0;i<m;++i){
        int u,v,t;
        cin>>u>>v>>t;
        q.push({u,v,t});
    }
    int k=n-1,res=-1;
    while(k--){
        auto t=q.top();q.pop();
        if(find(t.u)==find(t.v)){
            k++;
            continue;
        }
        fa[find(t.u)]=find(t.v);
        res=max(res,t.w);
    }
    cout<<res<<endl;

    return 0;
}




//201809-1

#include <bits/stdc++.h>
using namespace std;

const int N=1010;
int a[N];
int main()
{
	int n;
	cin>>n;
	for(int i=1;i<=n;++i)
		cin>>a[i];
	cout<<(a[1]+a[2])/2<<' ';
	for(int i=2;i<n;++i)
		cout<<(a[i-1]+a[i]+a[i+1])/3<<' ';
	cout<<(a[n-1]+a[n])/2;
	return 0;
}

//201809-2

#include <bits/stdc++.h>
using namespace std;
const int N=1000010;
int diff[N];
int n;

int main()
{

	cin>>n;
	for(int i=0;i<2*n;++i){
		int l,r;
		cin>>l>>r;
		diff[l]++,diff[r]--;
	}
	int res=0,t=0;
	for(int i=0;i<N;++i){
		t+=diff[i];
		if(t==2)
			++res;
	}
	cout<<res<<endl;

	return 0;
}




// 201809-3 元素选择器

//一维数组形式的树
//由于数据量非常小，树的关闭暴力遍历即可

#include <bits/stdc++.h>
using namespace std;

const int N = 110;
struct node {
   string lable, id;
   int level;
}tree[N];
unordered_map<string, int> mp;
set<int> res;
int n, m;
void to_lower(string& s) {
   transform(s.begin(), s.end(), s.begin(), ::tolower);
}
void dfs(int u, int fa, int line,vector<string>& s) {
   if (u == s.size()) {
       res.insert(line-1);     // !!!!!!!!!!!!!!!
       return;
   }
   if (line > n) return;       // !!!!!!!!!!!!
   for (int j = line; j <= n; ++j) {
       auto t = tree[j];
       if (t.level <= fa) return;      // 该分支完结   !!!!!!!!!!!!!!!!！！！！！！！！！！！
       if (t.lable == s[u] || t.id == s[u]) dfs(u + 1, t.level, j + 1,s);       // ！！！！！！！！！！！！！！
   }

}
int main() {

   cin >> n >> m;
   getchar();

   for (int i = 1; i <= n; ++i) {
       string s;
       getline(cin, s);
       int cnt = 0;
       while (s[cnt] == '.')++cnt;
       stringstream ss(s.substr(cnt));
       ss >> tree[i].lable >> tree[i].id; // 111111111111111111,id可以读不到，不会阻塞到这里
       to_lower(tree[i].lable);
       tree[i].level = cnt / 2;
   }
   for (int i = 1; i <= m; ++i) {
       string s;
       vector<string> str;
       res.clear();
       getline(cin, s);
       stringstream ss(s);
       while (ss >> s) {
           if (s[0] != '#') to_lower(s);       // !!!!!!!!!!!!!!
           str.push_back(s);
       }
       dfs(0, -1, 1,str);      // !!!!!!!!!!!!!
       cout << res.size();
       for (auto c : res) cout << ' ' << c;
       cout << endl;
   }
   return 0;
}


// 201809-4 再卖菜
// dfs+记忆化
#include <iostream>
using namespace std;
static const int N=310;
int n,a[N],b[N];
bool vis[N][N][N];      // vis[x][y][z]  第一天第x-1个商店价格为y同时第x-2个商店价格为z
void dfs(int d)
{
    if(vis[d][b[d-1]][b[d-2]]) return;      
    vis[d][b[d-1]][b[d-2]]=true;
    if(d==n+1)
    {
        if(a[n]!=(b[n-1]+b[n])/2)
            return;
        for(int i=1;i<=n;++i) cout<<b[i]<<' ';
        exit(0);        // !!!!!!
    }
    int L=3*a[d-1]-b[d-1]-b[d-2];       // ！！！！！！！！！！！！！！
    for(b[d]=L;b[d]<=L+2;b[d]++)        // ！！！！！！！！！！
        if(b[d]>0)dfs(d+1);
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin>>n;
    for(int i=1;i<=n;i++)
        cin>>a[i];
    for(b[1]=1;b[1]<=2*a[1];b[1]++)     // 注意边界         [1,2*a[1]]
        for(b[2]=2*a[1]-b[1];b[2]<=2*a[1]+1-b[1];b[2]++)    // [2*a[1]-b[1],2*a[1]+1-b[1]
            if(b[2]>0) dfs(3);
    return 0;
}



// 	201803-1

#include <iostream>
using namespace std;

int main()
{
	int res=0,last=1;
	int i=0;
	cin>>i;
	while(i){
		if(i==1) last=1;
		else {
			if(last==1) last=2;
			else last+=2;
		}
		res+=last;
		cin>>i;
	}
	cout<<res<<endl;


	return 0;
}



//201803-2

//模拟，数量不大，可以模拟

#include <bits/stdc++.h>
using namespace std;
const int N=110;
int n,L,t;
int origin[N],order[N],pos[N],dire[N];
int main()
{
	cin>>n>>L>>t;
	for(int i=1;i<=n;++i) {
		cin>>origin[i];
		dire[i]=1;
	}
	memcpy(order+1,origin+1,sizeof(int)*n);
	sort(order+1,order+1+n);
	memcpy(pos+1,order+1,sizeof(int)*n);

	pos[0]=pos[n+1]=-2;
	while(t--){
		for(int i=1;i<=n;++i){
			pos[i]+=dire[i];    // 走一步
			if(pos[i]==0 || pos[i]==L) dire[i] =- dire[i];    // 碰墙
			if(pos[i]==pos[i-1]) dire[i] = -dire[i];  // 和左边小球碰撞
			if(pos[i]==pos[i+1]+dire[i+1]) dire[i] = -dire[i];  // 和右边小球碰
		}
	}
	// 由于球之间相互碰撞改变方向，故各个球之间的相对位置不变(也可以结构体，变换排序规则也可以）
	for(int i=1;i<=n;++i)
		printf(" %d"+!(i-1),pos[lower_bound(order+1,order+1+n,origin[i])-order]);
	return 0;
}

//巧法
// 两球碰撞的话（不包括碰撞边缘的情况），交换速度，两个球接下来走的路本来应该是对方球要走的路，
// 	其实可以看做没有碰撞，即两个球继续沿着原来的路走下去。且球的相对位置不会改变。
#include <bits/stdc++.h>
using namespace std;
const int N=110;
int n,L,t;
int origin[N],order[N],pos[N];
int main()
{
	cin>>n>>L>>t;
	for(int i=1;i<=n;++i)
		cin>>origin[i];

	memcpy(order+1,origin+1,sizeof(int)*n);
	sort(order+1,order+1+n);
	for(int i=1;i<=n;++i){
		pos[i]=(origin[i]+t)%(2*L) > L ? 2*L - (origin[i]+t)%(2*L) : (origin[i]+t)%(2*L);

	}
	sort(pos+1,pos+1+n);
	// 由于球之间相互碰撞改变方向，故各个球之间的相对位置不变(也可以结构体，变换排序规则也可以）
	for(int i=1;i<=n;++i)
		printf(" %d"+!(i-1),pos[lower_bound(order+1,order+1+n,origin[i])-order]);
	return 0;
}


// 201803-3 URL映射
// 使用getline(ss,s,'/')将规则和url进行拆分，同时需要注意规则和url之后是否有‘/’，这也是区别的
//      故人为最后进行‘/’的补充

#include <bits/stdc++.h>
using namespace std;

const int N=110;
vector<string> rule[N],url;
string name[N];
string query;

int n,m;
void prase(int u,string s){
 stringstream ss(s);
 getline(ss,s,'/');      // !!!!!!!!!!!!!! 去除首次读入的空行
 while(getline(ss,s,'/')){
     rule[u].push_back(s);
 }
}
// 1字符串  0 数字
int check_type(string s){
 for(auto c:s){
     if(!isdigit(c)) return 1;
 }
 return 0;
}
bool check(int u){
 vector<string> res;
 int j=0;
 for(int i=0;i<url.size();++i){
     int t=check_type(url[i]);
     if(j>=rule[u].size()) return false;      // !!!!!!! 注意边界

     if(rule[u][j]=="<int>"){
         if(t!=0) return false;
         res.push_back(to_string(atoi(url[i].c_str())));//     !!!!!!!! 去前导零
         ++j;
     }else if(rule[u][j]=="<str>" && url[i]!="/"){        // !!!!!!!!!!!!!!b[i]!="/"
        // 用于匹配一段字符串，注意字符串里不能包含斜杠
         res.push_back(url[i]);
         ++j;
     }else if(rule[u][j]=="<path>" && url[i]!="/"){       // !!!!!!!!!!!!!!b[i]!="/"
        // <path> 的前面一定是斜杠，后面一定是规则的结束
         string t="/";
         for(int k=0;k<i;++k) t+=url[k]+'/';
         res.push_back(query.substr(t.size()));     // 截取之后所有字符
         ++j;
         break;     // ！！！！！！
     }else{
         if(url[i]!=rule[u][j]) return false;
         ++j;
     }

 }
 if(j<rule[u].size()) return false;

 cout<<name[u];
 for(auto c:res) cout<<' '<<c;
 cout<<endl;
 return true;
}
int main()
{
 cin>>n>>m;
 for(int i=1;i<=n;++i){
     string s;
     cin>>s;
     prase(i,s);
     if(s.back()=='/') rule[i].push_back("/");      //!!!!!!!!!!! 手动添加最后的路径
     cin>>name[i];
 }
 for(int i=1;i<=m;++i){
     string s;
     cin>>query;
     url.clear();

     stringstream ss(query);
     getline(ss,s,'/');              //  !!!!!!!!!!!!!! 去除首次读入的空行
     while(getline(ss,s,'/')){
         url.push_back(s);
     }
     if(query.back()=='/') url.push_back("/"); // !!!!!!!!!!!!!!!!!!

     int j;
     for(j=1;j<=n;++j){
         if(check(j)) break;
     }
     if(j>n) cout<<404<<endl;
 }

 return 0;
}


// 201803-4 棋局评估
// 博弈论对抗搜索
//Alice获胜时得到分是正分，Bob获胜时得到的分是负分，由于两个人都非常聪明，所以在任意局面下：
//若当前步是Alice下，那么Alice会使自己可以得到的分数最大：
//若当前步是Bob下，那么Bob会使自己可以得到的分数最大，负分表示就是最小；

//根据以上的得分策略，枚举所有的下棋情况，具体来说：
//①设DFS(i)为当前局面下，第i(i==0或1)个人下的时候，该人能获得的"最大值"；
//②若当前步是Alice下，那么遍历Alice能下的所有位置，返回后续所有结果中最终局面的最大值；
//③若当前步时Bob下，那么遍历Bob能下的所有位置，返回后续所有结果中最终局面的最小值；
//④这里数据比较小，最多也就9！次，直接搜索即可


#include <bits/stdc++.h>
using namespace std;
const int INF=1e9;

int g[5][5];

// x赢不赢
bool judge(int x){
    if(g[2][2]==x && (g[1][1]==x && g[3][3]==x || g[1][3]==x && g[3][1]==x)) return true;
    for(int i=1;i<=3;++i){
        if(g[1][i]==x && g[2][i]==x && g[3][i]==x) return true;
        if(g[i][1]==x && g[i][2]==x && g[i][3]==x) return true;
    }
    return false;
}
int f(){
    int res=0;
    for(int i=1;i<=3;++i)
        for(int j=1;j<=3;++j) res+=g[i][j]==0;

    if(judge(1)) return res+1;
    if(judge(2)) return -res-1;
    if(res==0) return 0;        // 平局
    return INF;     // 未结束
}
// 当前由u来下字，返回当前局面的最终分数，当前局面的最终分数的正负与谁来下无关
// Alice希望分数最大，Bob希望分数最小
int dfs(bool u){
    int t=f();
    if(t!=INF) return t;
    
    int res=0;
    if(u){      // true X  false O
        // Alice返回接下来所有可能局面的最大值
        res=-INF;
        for(int i=1;i<=3;++i)
            for(int j=1;j<=3;++j){
                if(!g[i][j]){
                    g[i][j]=1;
                    res=max(res,dfs(false));
                    g[i][j]=0;
                }
            }
    }else{
        // Bob返回接下来所有可能局面的最小值
        res=INF;
        for(int i=1;i<=3;++i)
            for(int j=1;j<=3;++j){
                if(!g[i][j]){
                    g[i][j]=2;
                    res=min(res,dfs(true));
                    g[i][j]=0;
                }
            }
    }
    return res;
}

int main()
{
    int T;
    cin>>T;

    while(T--){
        for(int i=1;i<=3;++i)
            for(int j=1;j<=3;++j)
                cin>>g[i][j];
        cout<<dfs(true)<<endl;
    }

    return 0;
}


//201712-1

#include <bits/stdc++.h>
using namespace std;
const int N=1010;
int a[N];
int main()
{
	int n;
	cin>>n;
	for(int i=0;i<n;++i) cin>>a[i];
	sort(a,a+n);
	int res=INT_MAX;
	for(int i=1;i<n;++i){
		res=min(res,a[i]-a[i-1]);
	}
	cout<<res<<endl;
	return 0;
}


//201712-2
//循环单链表
#include <bits/stdc++.h>
using namespace std;

const int N=1010;
int ne[N];
int main()
{
   int n,k;
   cin>>n>>k;
   for(int i=1;i<n;++i) ne[i]=i+1;
   ne[n]=1;
   int rest=n,p=n,num=1;
   while(rest>1){
       if(num%10==k || num%k==0){
           --rest;
           ne[p]=ne[ne[p]];
       }else
           p=ne[p];
       ++num;
   }
   cout<<ne[p];
   return 0;
}



// 201712-3 Crontab

#include <bits/stdc++.h>
using namespace std;

int month[2][13]={{0,31,28,31,30,31,30,31,31,30,31,30,31},
                 {0,31,29,31,30,31,30,31,31,30,31,30,31}};
unordered_map<string,int> mp;

struct node{
 int y,m,d,h,mi,w;
 node() =default;
 node(string s){
     sscanf(s.c_str(),"%04d%02d%02d%02d%02d",&y,&m,&d,&h,&mi);
 }
 string to_string(){
     char s[20];
     sprintf(s,"%04d%02d%02d%02d%02d",y,m,d,h,mi);
     return s;
 }
 bool operator==(const node& a) const{
     return y==a.y && m==a.m && d==a.d && h==a.h && mi==mi;
 }
 node &operator++(){
     if(++mi==60){
         mi=0;
         if(++h==24){
             h=0;
             ++d;
             w=(w+1)%7;
             int t=isr(y);
             if(d>month[t][m]){
                 d=1;
                 if(++m>12){
                     m=1;
                     ++y;
                 }
             }
         }
     }
     return *this;
 }
 bool isr(int y){
     return y%400==0 || y%4==0 && y%100!=0;
 }
};

struct Task{
 // 分 时 天 月 星 命令
 bool mi[60],h[24],d[32],m[13],w[7];
 string cmd;
 bool check(node& t){
     return mi[t.mi] && h[t.h] && d[t.d] && m[t.m] && w[t.w];
 }
}task[20];

node L,R;

void init(){
 string keys[] = {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
                   "sun", "mon","tue","wed", "thu","fri","sat"};
   int values[] = {1,2,3,4,5,6,7,8,9,10,11,12,0,1,2,3,4,5,6};
   for (int i=0;i<=18;i++) mp[keys[i]]=values[i];
}
int to_num(string s){
 if(isdigit(s[0])) return atoi(s.c_str());
 transform(s.begin(),s.end(),s.begin(),::tolower);
 return mp[s];
}
void work(string s,bool st[],int len){
 stringstream ss(s);
 while(getline(ss,s,',')){      // 逗号
     int p;
     if(s=="*"){        // 任何取值
         for(int i=0;i<len;++i) st[i]=true;
     }else if((p=s.find('-'))!=-1){         // 一段连续的取值范围
         for(int i=to_num(s.substr(0,p));i<=to_num(s.substr(p+1));++i) st[i]=true;
     }else{                 // 特定值
         st[to_num(s)]=true;
     }
 }
}
int main()
{
 init();
 int n;
 string s1,s2;
 cin>>n>>s1>>s2;
 L=node(s1),R=node(s2);
 
 // 分 时 天 月 星 命令
 for(int i=0;i<n;++i){
     string mi,h,d,m,w;
     cin>>mi>>h>>d>>m>>w>>task[i].cmd;
     work(mi,task[i].mi,60);
     work(h,task[i].h,24);
     work(d,task[i].d,32);
     work(m,task[i].m,13);
     work(w,task[i].w,7);
 }
 
 
 node T=node("197001010000");
 T.w=4;
 while(!(T==L)){
     ++T;
 }
 
 while(!(T==R)){
     for(int i=0;i<n;++i){
         if(task[i].check(T)) cout<<T.to_string()<<' '<<task[i].cmd<<'\n';
     }
     ++T;
 }
 return 0;
}




// 201712-4 行车路线
#include <bits/stdc++.h>
using namespace std;

struct path{
    int u,t,d;      // u是边的终点，t是最近连续走的小路的长度，d是节点1到当前节点的疲劳度
    path(int uu,int tt,int dd):u(uu),t(tt),d(dd){}
    bool operator<(const path& a) const {
        return d>a.d;
    }
};
const int INF=1e6+1;    // 因为答案不超过1e6,故连续的小路长度不超过1e3！！！！！！！！！！！！！！！！！！！
const int N=510;
vector<path> he[N];
int dist[N][1010];        // dist[i][j]是节点1到达节点i且最后连续走了j距离的小路
bool vis[N][1010];
int n,m;
priority_queue<path> q;

void dijkstra(){
    fill(dist[0],dist[0]+N*1010,INF);   // !!!!!!! 剪枝
    dist[1][0]=0;

    q.push({1,0,0});
    while(!q.empty()){
        auto t=q.top();
        q.pop();
        if(vis[t.u][t.t]) continue;
        vis[t.u][t.t]=true;
        
        for(auto i:he[t.u]){
            if(i.t==0){     // 大路
                if(dist[i.u][0]>t.d+i.d){
                    dist[i.u][0]=t.d+i.d;
                    q.push({i.u,0,t.d+i.d});
                }
            }else{      // 小路
                int len=t.t+i.d;
                if(len>1000) continue;      // !!!!!!!!!!!!!!!!!!!! 注意边界
                if(dist[i.u][t.t+i.d]>t.d+(t.t+i.d)*(t.t+i.d)-t.t*t.t){//！！！！！！
                    dist[i.u][t.t+i.d]=t.d+(t.t+i.d)*(t.t+i.d)-t.t*t.t;
                    q.push({i.u,t.t+i.d,dist[i.u][t.t+i.d]});
                }
            }
        }
    }
    int res=INT_MAX;
    for(int i=0;i<1010;++i) res=min(res,dist[n][i]);
    cout<<res<<endl;
    
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    
    cin>>n>>m;
    for(int i=0;i<m;++i){
        int t,a,b,c;
        cin>>t>>a>>b>>c;
        he[a].push_back({b,t,c});
        he[b].push_back({a,t,c});
    }
    dijkstra();
    
    return 0;
}




//201709-1


#include <bits/stdc++.h>
using namespace std;

int main()
{
	int n;
	cin>>n;

	n/=10;
	int k=n/5;
	int sum=k*5+k*2;
	k=n%5;
	sum+=k;
	if(k>=3) sum+=1;
	cout<<sum<<endl;
	return 0;
}



//201709-2

#include <bits/stdc++.h>
using namespace std;

const int N=1010;
int n,k;
int a[N];
struct node{
	int id,t,st;
};
int main()
{
	cin>>n>>k;
	for(int i=1;i<=n;++i) a[i]=i;

	vector<node> times;
	for(int i=1;i<=k;++i){
		int id,start,t;
		cin>>id>>start>>t;
		// 1取走，0还
		times.push_back({id,start,1});
		times.push_back({id,start+t,0});
	}
	// 先按照时间点排序，再按照还-取的顺序排序，最后按照编号排序
	sort(times.begin(),times.end(),[](const node& a,const node& b){
		if(a.t!=b.t) return a.t<b.t;
       if(a.st!=b.st) return a.st<b.st;
		return a.id<b.id;
	});
	
	for(auto i:times){
		if(i.st==0){        // 还
			for(int j=1;j<=n;++j){
				if(a[j]==-1){
					a[j]=i.id;
					break;
				}
			}
		}else{      // 取
			for(int j=1;j<=n;++j){
				if(a[j]==i.id){
					a[j]=-1;
					break;
				}
			}
		}
	}
	
	for(int i=1;i<=n;++i)
		printf(" %d"+!(i-1),a[i]);
		
	return 0;
}

// 201709-3 JSON查询

//定义node作为值，先去掉所有空格，在递归构造对象

#include <bits/stdc++.h>
using namespace std;
struct node{
 int type;   // 0 值 1 对象
 string value;
 node()=default;
 node(string v):value(v),type(0){}
 unordered_map<string,node*> mp;
};
int n,m;
string str;
node* prase(int& u){
 if(str[u]=='{'){
     node* t=new node;
     t->type=1;
     if(str[u+1]!='}'){  // 不是空对象
         while(str[u]!='}'){
             string key;
             u+=2;       // 从首个字符开始!!!!!!!!!!!!!!!!!!!
             while(str[u]!='"'){        // 解析的是键
                 if(str[u]=='\\') u++;
                 key+=str[u++];
             }
             u+=2;   // 跳过":!!!!!!!!!!!!!!!!!!!!!!!!!
             t->mp[key]=prase(u);
         }
     }
     u++;    // 跳过}
     return t;
 }else if(str[u]=='"'){         // 解析的是值
     string value;
     u++;           // !!!!!!!!!!!!!!!!!!!!!!!
     while(str[u]!='"'){
         if(str[u]=='\\') u++;
         value+=str[u++];
     }
     u++;           // !!!!!!!!!!!!!!!!!!
     return new node(value);
 }

}
void query(node* obj,string p){
 stringstream s(p);
 while(getline(s,p,'.')){
     obj=obj->mp[p];
     if(obj==nullptr){
         cout<<"NOTEXIST"<<endl;
         return;
     }
 }
 if(obj->type==1){
     cout<<"OBJECT"<<endl;
     return;
 }
 cout<<"STRING "<<obj->value<<endl;

}
int main()
{
 cin>>n>>m;
 getchar();
 string line;
 for(int i=0;i<n;++i){
     getline(cin,line);
     string s;
     for(int i=0;i<line.size();++i){
         if(line[i]==' ') continue;      // !!!!!! 空格直接跳，不会出现在字符串中间
         s+=line[i];
     }
     str+=s;
 }

 int u=0;
 node* obj=prase(u);

 for(int i=0;i<m;++i){
     string q,s;
     cin>>q;
     query(obj,q);
 }

 return 0;
}


//  201709-4 通信网络
// 题意：n个节点，m条单向边，如果有一个节点能够发送信息的节点和能够接收信息的节点个数再加上自己为n的话，则该节点能够知晓全部的n个节点，问这样的节点有多少
//floyd
#include <bits/stdc++.h>
using namespace std;
const int N=1010;

int dist[N][N];
int cnt[N];
int main()
{
    int n,m;
    cin>>n>>m;
    memset(dist,0x3f,sizeof dist);
    for(int i=0;i<=n;++i) dist[i][i]=0;
    
    for(int i=1;i<=m;++i){
        int x,y;
        cin>>x>>y;
        dist[x][y]=min(dist[x][y],1);
    }
    
    for(int k=1;k<=n;++k)
        for(int i=1;i<=n;++i)
            for(int j=1;j<=n;++j){
                dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
            }
            
    for(int i=1;i<=n;++i)
        for(int j=i;j<=n;++j){
            if(dist[i][j]<0x3f3f3f3f || dist[j][i]<0x3f3f3f3f){
                if(i!=j)        // !!!!!!!!!!!!
                    ++cnt[i],++cnt[j];
                else
                    ++cnt[i];
            }
        
        }
    int res=0;
    for(int i=1;i<=n;++i)
        res+=cnt[i]==n;
        
    cout<<res<<endl;

    return 0;
}



//201703-1

#include <bits/stdc++.h>
using namespace std;


int main()
{
	int n,k;
	cin>>n>>k;
	int sum=0,cnt=0;
	for(int i=1;i<=n;++i){
		int a;
		cin>>a;
		sum+=a;
		if(sum>=k){
			++cnt;
			sum=0;
		}
	}
	if(sum) ++cnt;
	cout<<cnt<<endl;
	return 0;
}


//201703-2

#include <bits/stdc++.h>
using namespace std;
const int N=1010;
int n,m;
int a[N];
int main()
{
	cin>>n>>m;
	for(int i=1;i<=n;++i) a[i]=i;
	for(int i=1;i<=m;++i){
		int p,q;
		cin>>p>>q;
		int pos=find(a+1,a+1+n,p)-a;
		if(q>0){
			for(int i=pos;i<pos+q;++i) a[i]=a[i+1];
			a[pos+q]=p;
		}else{
			for(int i=pos;i>pos+q;--i) a[i]=a[i-1];
			a[pos+q]=p;
		}
		
	}
   for(int i=1;i<=n;++i) printf(" %d"+!(i-1),a[i]);
	return 0;
}



// 201703-3 Markdown
// 先统一对区块进行处理，再统一处理行内
#include <bits/stdc++.h>
using namespace std;

int main()
{
 string line,res;
 vector<string> ul,pp;
 while(getline(cin,line)){
     if(line.empty()){      // 结算
         if(!ul.empty()){
             res+="<ul>\n";
             for(auto c:ul)
                 res+="<li>"+c+"</li>"+'\n';
             res+="</ul>\n";
             ul.clear();
         }else if(!pp.empty()){
             res+="<p>";
             for(auto c:pp)
                 res+=c+'\n';
             res.pop_back();
             res+="</p>\n";
             pp.clear();
         }
         // ！！！！！！！！！不需要换行，记住以输入结果为准
     } else if(line[0]=='#'){       // 标题
         int t=0;
         while(line[t]=='#') ++t;
         int p=t;
         while(line[p]==' ') ++p;
         line=line.substr(p);           // 去除标题后多余空格，只保留标题
         res+="<h"+to_string(t)+">"+line+"</h"+to_string(t)+">\n";
     } else if(line[0]=='*'){       // 列表
         int t=0;
         while(line[t]=='*') ++t;
         int p=t;
         while(line[p]==' ') ++p;
         ul.push_back(line.substr(p));
     }else{                 // 段落
         pp.push_back(line);
     }
 }
 // ！！！！！！！！！！！！！！！！！
 if(!ul.empty()){
     res+="<ul>\n";
     for(auto c:ul)
         res+="<li>"+c+"</li>"+'\n';
     res+="</ul>\n";
     ul.clear();
 }else if(!pp.empty()){
     res+="<p>";
     for(auto c:pp)
         res+=c+'\n';
     res.pop_back();
     res+="</p>\n";
     pp.clear();
 }
 res.pop_back();        // 去除最后换行！！！！！！！！！！！！！！！！
 
 
 // 处理行内
 string::size_type p=0,t=0,ne=0;
 while((p=res.find("_",p))!=res.npos){      // 强调
     t=res.find("_",p+1);        // !!!!!!!!!!!!!!!!  p+1
     string sub=res.substr(p+1,t-p-1);
     res.replace(p,t-p+1,"<em>"+sub+"</em>");       // ！！！！！！！！！！！
 }


 p=0,t=0;
 while((p=res.find("[",p))!=res.npos){      // 超链接
     t=res.find("]",p);
     string sub1=res.substr(p+1,t-p-1);
     
     ne=res.find(")",t);
     string sub2=res.substr(t+2,ne-t-2);
     res.replace(p,ne-p+1,"<a href=\""+sub2+"\">"+sub1+"</a>");
 }

 cout<<res<<endl;
 return 0;
}




// 201703-4 地铁修建
// dijkstra
// 题意：n个节点，修m条边，m条边可以同时修，问最快修多久
#include <bits/stdc++.h>
using namespace std;
const int N=100010;
struct node{
    int u,d;
    node(int uu,int dd):u(uu),d(dd){}
    bool operator<(const node& a) const {
        return d>a.d;
    }
};
int dist[N];
int n,m;
vector<node> he[N];
priority_queue<node> q;
void dijkstra(){
    memset(dist,0x3f,sizeof dist);
    q.push({1,-1});

    while(!q.empty()){
        auto t=q.top();
        q.pop();
        
        for(int i=0;i<he[t.u].size();++i){
            auto v=he[t.u][i];
            if(max(t.d,v.d)<dist[v.u]){
                dist[v.u]=max(t.d,v.d);
                q.push({v.u,dist[v.u]});
            }
        }
    }
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    
    cin>>n>>m;
    for(int i=0;i<m;++i){
        int a,b,c;
        cin>>a>>b>>c;
        he[a].push_back({b,c});
        he[b].push_back({a,c});
    }
    dijkstra();
    
    cout<<dist[n];
    return 0;
}



//201612-1

#include <bits/stdc++.h>
using namespace std;

const int N=1010;
int a[N];
int main()
{
   int n;
   cin>>n;
   for(int i=0;i<n;++i) cin>>a[i];
   sort(a,a+n);
   int l=n/2,r=n/2,num=a[n/2];
   while(a[l]==num) --l;
   while(a[r]==num) ++r;
   if(l+1==n-r) cout<<num;
   else cout<<-1;
   return 0;
}


// 201612-2

#include <bits/stdc++.h>
using namespace std;

int a[]={0,1500,4500,9000,35000,55000,80000,INT_MAX};
double k[]={0.03,0.1,0.2,0.25,0.3,0.35,0.45};
int main()
{
	int T;
	cin>>T;

	int sum;
	if(T>3500){
		sum=3500;
		T-=3500;
		for(int i=0;i<7;++i){
			int c=(a[i+1]-a[i])*(1-k[i]);
			if(T>=c){
				sum+=a[i+1]-a[i];
				T-=c;
			}else{
				sum+=T/(1-k[i]);
				T=0;
				break;
			}
		}
	}else
		sum=T;

	cout<<sum<<endl;
	return 0;
}



//  201612-3 权限查询
// 每位用户具有若干角色，每种角色具有若干权限。
// 权限分为分等级权限和不分等级权限两大类。
// 分等级权限由权限类名和权限等级构成，中间用冒号“:”分隔。
// 查询种类：不分等级权限的查询，分等级权限的带等级查询，分等级权限的不带等级查询
#include <bits/stdc++.h>
using namespace std;

struct node{
 string pri;        // 权限
 int lev;        // -1标记为无等级，1~10标记为权限等级，比题目的0~9多1，数字越大表示权限等级越高。由于unordered_map不存在是，默认为0，0认为权限不存在
 node(){
     lev=-1;
  }
 node(string p,int l):pri(p),lev(l){}
};
int p,r,u,q;

unordered_map<string,vector<node>> role;        // 角色对应的权限
unordered_map<string,unordered_map<string,int>> user;       // 记录用户权限对应的最高权限
node prase(string& s){
 int le;
 if(s.find(':')==s.npos) le=-1;
 else {
     le=s.back()-'0';
     s.pop_back();s.pop_back();
     le++;
 }
 return {s,le};
}
int main()
{
 cin>>p;
 for(int i=0;i<p;++i){      // p个没用的数据，应为输入保证合法
     string s;
     cin>>s;
 }
 cin>>r;
 for(int i=0;i<r;++i){
     string ro,pr;
     int s;
     cin>>ro>>s;
     for(int j=0;j<s;++j){
         cin>>pr;
         auto res=prase(pr);
         role[ro].push_back(res);
     }
 }
 cin>>u;
 for(int i=0;i<u;++i){
     string us,ro;
     int t;
     cin>>us>>t;
     for(int j=0;j<t;++j){
         cin>>ro;
         for(auto c:role[ro]){
             if(c.lev==-1)
                 user[us][c.pri]=-1;
             else
                 user[us][c.pri]=max(user[us][c.pri],c.lev);
         }
     }
     
 }
 cin>>q;
 for(int i=0;i<q;++i){
     string us,pr;
     cin>>us>>pr;
     auto res=prase(pr);
     if(res.lev==-1){    // 无等级或不带等级
        // 不分等级权限的查询：如果权限本身是不分等级的，则查询时不指定等级，返回是否具有该权限
        // 分等级权限的不带等级查询：如果权限本身分等级，查询不带等级，则返回具有该类权限的等级；如果不具有该类的任何等级权限，则返回“否”。
         if(user[us][res.pri]==0) cout<<"false"<<endl;   // 无该权限
         else if(user[us][res.pri]==-1) cout<<"true"<<endl;  // 无等级
         else{
             cout<<user[us][res.pri]-1<<endl;
         }
     }else{      // 有等级权限
        // 分等级权限的带等级查询：如果权限本身分等级，查询也带等级，则返回是否具有该类的该等级权限；
         if(user[us][res.pri]<res.lev) cout<<"false"<<endl;
         else cout<<"true"<<endl;
     }
 }
 return 0;
}


//  201612-4  压缩编码

//虽然外表套了个壳，但实际是石子合并问题，区间dp
// 按哈夫曼树编码和按字典序升序，和两者的相同点: 每次合并两个节点, 要求最小代价
// 两者的不同点: 哈夫曼树随意合并, 字典序需要按照顺序合并, 也就是只能合并相邻的两个点
// 只能合并相邻的两个点而且要最小代价，那就是石子合并问题
//下面的代码中表面上看没有考虑每个单词的编码长度问题，但实践上两两合并过程和哈夫曼树是一样的，
//     参与合并一次，即说明编码长度+1
#include <bits/stdc++.h>
using namespace std;

const int N=1010;
int t[N],f[N][N];
int main()
{
    int n;
    cin>>n;
    for(int i=1;i<=n;++i) cin>>t[i],t[i]+=t[i-1];       // 前缀和
    
    for(int len=2;len<=n;++len)     // 长度
        for(int i=1;i<=n-len+1;++i)     // 左端点
        {
            int j=i+len-1;      //  右端点
            f[i][j]=1e8;        // f[i][i]是全局变量，自动为0
            for(int k=i;k<j;++k)
                f[i][j]=min(f[i][j],f[i][k]+f[k+1][j]+t[j]-t[i-1]);

        }

    
    cout<<f[1][n];
    
    return 0;
}




//201609-1


#include <bits/stdc++.h>
using namespace std;


int main()
{
	int n;
	cin>>n;
	int p,c,res=-1;
	cin>>p;
	for(int i=1;i<n;++i){
		cin>>c;
		res=max(res,abs(c-p));
		p=c;
	}
	cout<<res<<endl;
	return 0;
}

//201609-2


#include <bits/stdc++.h>
using namespace std;
bool vis[20][5];

int main()
{
	int n;
	cin>>n;
	while(n--){
		int t;
		cin>>t;
		int i=0;
		for(;i<20;++i){
			int j=0;
			for(;j<=5-t;++j){
				int k=0;
				for(;k<t;++k){
					if(vis[i][j+k])
						break;
				}
				if(k==t) break;
			}

			if(j<=5-t){
//				for(int p=j;p<j+t;++p){
//					cout<<i*5+p+1<<' ';
//					vis[i][p]=true;
//				}

				int end=j+t;
				for(;j<end;++j){
					cout<<i*5+j+1<<' ';
					vis[i][j]=true;
				}
				break;
			}
		}

		if(i==20){
			for(i=0;i<20;++i){
				for(int j=0;j<5;++j){
					if(!vis[i][j]){
						vis[i][j]=true;
						--t;
						cout<<i*5+j+1<<' ';
					}
					if(!t) break;
				}
				if(!t) break;
			}
		}
		cout<<endl;
	}
	return 0;
}



// 201609-3 炉石传说
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> PII;

vector<PII> a[2];

int main()
{
 int n;
 cin>>n;
 int i=0;
 //玩家各控制一个英雄，游戏开始时，英雄的生命值为 30，攻击力为 0。
 a[0].push_back({30,0}),a[1].push_back({30,0});

 while(n--){
     string op;
     cin>>op;
        // 位置<position>召唤一个生命值为<health>、攻击力为<attack>的随从
         if(op=="summon"){
         int p,at,h;
         cin>>p>>at>>h;       // 输入顺序要搞好
         a[i].insert(a[i].begin()+p,make_pair(h,at));
     }else if(op=="attack"){
        // attack <attacker> <defender>：当前玩家的角色<attacker>攻击对方的角色 <defender>
         int x,y;
         cin>>x>>y;
         int my=i,your=i==0?1:0;
         a[my][x].first-=a[your][y].second;
         a[your][y].first-=a[my][x].second;
         // 英雄死亡之后对随从的位置没有影响 !!!!!!!!!!!
         if(a[my][x].first<=0 && x>0) a[my].erase(a[my].begin()+x);
         if(a[your][y].first<=0 && y>0) a[your].erase(a[your].begin()+y);
     }else{
         i=i==0?1:0;
     }
 }
 int v;
 if(a[0][0].first>0 && a[1][0].first>0) v=0;
 else if(a[0][0].first>0) v=1;
 else v=-1;

 cout<<v<<endl;
 cout<<a[0][0].first<<endl<<a[0].size()-1<<' ';
 for(i=1;i<a[0].size();++i) cout<<a[0][i].first<<' ';
 cout<<endl;


 cout<<a[1][0].first<<endl<<a[1].size()-1<<' ';
 for(i=1;i<a[1].size();++i) cout<<a[1][i].first<<' ';
 
}


// 201609-4 交通规划
#include <iostream>
#include <cstring>
#include <queue>
#include <stack>
using namespace std;
typedef pair<int, int> PII;
const int N=10010,M=200010;     // !!!!!!!!!!!!!!!!!!!!!!!!!!! 边是无向的
int head[N],ver[M],edge[M],ne[M],d[N];
int mi[N];
bool v[N];
int n,m,tot,res;

priority_queue<PII, vector<PII>, greater<PII>> q;        // 默认大根堆
void add(int x,int y,int z){
    // ver是边的终点，edge是边权值，next为下一条边
    ver[++tot]=y,edge[tot]=z,ne[tot]=head[x],head[x]=tot;
}

void dijkstra(){
    memset(d,0x3f,sizeof d);
    memset(mi,0x3f,sizeof mi);
    d[1]=0;
    mi[1]=0;
    
    q.push({0,1});
    while(q.size()){
        int x=q.top().second;q.pop();
        if(v[x]) continue;
        v[x]=1;
        res+=mi[x];
        for(int i=head[x];i;i=ne[i]){
            int y=ver[i],z=edge[i];
            if(d[y]> d[x] +z){
                d[y]=  d[x] +z;
                mi[y]=z;
                q.push({d[y],y});
            }else if(d[y]==  d[x] +z){
                if(z<mi[y]){
                    mi[y]=z;
                }
            }
        }
    }
    
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin>>n>>m;
    for(int i=0;i<m;++i){
        int x,y,c;
        cin>>x>>y>>c;
        add(x,y,c);
        add(y,x,c);
    }
    dijkstra();
    cout<<res<<endl;
    return 0;
}



//201604-1

#include <bits/stdc++.h>
using namespace std;

int a[1010];
int d[1010];
int main()
{
	int n;
	cin>>n;
	for(int i=1;i<=n;++i)
		cin>>a[i];

	for(int i=2;i<=n;++i)
		d[i]=a[i]-a[i-1];

	int res=0;
	for(int i=2;i<=n;++i)
		if(d[i]*d[i-1]<0) ++res;
	cout<<res<<endl;
	return 0;
}


//201604-2



#include <bits/stdc++.h>
#include <cstring>
using namespace std;

int a[20][10];      // 行开大点
int in[4][4];
int main()
{
	for(int i=0;i<15;++i)
		for(int j=0;j<10;++j)
			cin>>a[i][j];
   for(int i=0;i<4;++i)
		for(int j=0;j<4;++j)
			cin>>in[i][j];
	int t;
	cin>>t;
	t--;

	int i=1;
	for(;i<15;++i){        // 最终行往最下
		bool ok=true;
		for(int j=0;j<4;++j)
			for(int k=0;k<4;++k){
				if(in[j][k]+a[j+i][k+t]==2) ok=false;
				if(in[j][k]==1 && (j+i)>=15) ok=false;   // 4*4的1不要太往下，消失
			}
		if(!ok) break;
	}
	i--;
	for(int j=0;j<4;++j)
		for(int k=0;k<4;++k)
			a[j+i][k+t]+=in[j][k];

	for(int j=0;j<15;++j){
		for(int k=0;k<10;++k){
			cout<<a[j][k]<<' ';
		}
		cout<<endl;
	}
	return 0;
}


// 201604-3 路径解析 

#include <iostream>
#include <string>
#include <vector>
using namespace std;

string path;
int n;

void parse(string s){
    // 若路径为空字符串，则正规化操作的结果是当前目录。
     if(s.empty()) {
         cout<<path<<endl;
         return ;
     }
     
     char last=s[0];
     string t;
     t+=s[0];
     // 如果有多个连续的 / 出现，其效果等同于一个 /
     for(int i=1;i<s.size();++i){
         if(s[i]=='/' && last=='/') continue;
         t+=s[i];
         last=s[i];
     }
     
     vector<string> ans;
     int i=0;
     if(t[0]!='/') t=path+t;        // 没从根开始
     
     // 按照‘/’分割处理
     while(i<t.size()){     
         ++i;           // 跳过‘/'
         if(i>=t.size()) break;
         string tmp;
         while(t[i]!='/' && i<t.size()) tmp+=t[i++];
         ans.push_back(tmp);
     }
     vector<string> res;
     for(int i=0;i<ans.size();++i){
         if(ans[i]==".." && !res.empty()) res.pop_back();
         else if(ans[i]==".") continue;
         else res.push_back(res[i]);
     }
     for(int i=0;i<res.size();++i)
         cout<<'/'<<res[i];
     cout<<endl;

}
int main()
{
 cin>>n;
 cin>>path;
 getchar();
 while(n--){
     string s;
     cin>>s;
     parse(s);
 }
 return 0;
}



// 201609-4 交通规划 ❌考过了
#include <iostream>
#include <cstring>
#include <queue>
#include <stack>
using namespace std;
typedef pair<int, int> PII;
const int N=10010,M=200010;     // !!!!!!!!!!!!!!!!!!!!!!!!!!! 边是无向的
int head[N],ver[M],edge[M],ne[M],d[N];
int mi[N];
bool v[N];
int n,m,tot,res;

priority_queue<PII, vector<PII>, greater<PII>> q;        // 默认大根堆
void add(int x,int y,int z){
    // ver是边的终点，edge是边权值，next为下一条边
    ver[++tot]=y,edge[tot]=z,ne[tot]=head[x],head[x]=tot;
}

void dijkstra(){
    memset(d,0x3f,sizeof d);
    memset(mi,0x3f,sizeof mi);
    d[1]=0;
    mi[1]=0;
    
    q.push({0,1});
    while(q.size()){
        int x=q.top().second;q.pop();
        if(v[x]) continue;
        v[x]=1;
        res+=mi[x];
        for(int i=head[x];i;i=ne[i]){
            int y=ver[i],z=edge[i];
            if(d[y]> d[x] +z){
                d[y]=  d[x] +z;
                mi[y]=z;
                q.push({d[y],y});
            }else if(d[y]==  d[x] +z){
                if(z<mi[y]){
                    mi[y]=z;
                }
            }
        }
    }
    
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin>>n>>m;
    for(int i=0;i<m;++i){
        int x,y,c;
        cin>>x>>y>>c;
        add(x,y,c);
        add(y,x,c);
    }
    dijkstra();
    cout<<res<<endl;
    return 0;
}



//201512-1

#include <bits/stdc++.h>
using namespace std;

int main()
{
	long long n;
	cin>>n;
	int res=0;
	while(n){
		res+=n%10;
		n/=10;
	}
	cout<<res<<endl;
	return 0;
}


//201512-2
//+型模拟，6个位置
#include <bits/stdc++.h>
using namespace std;

int a[40][40];
bool vaild(int r,int c){
	if(a[r][c]==a[r+1][c] && a[r][c]==a[r+2][c] ) return true;
	if(a[r][c]==a[r-1][c] && a[r][c]==a[r-2][c] ) return true;
	if(a[r][c]==a[r-1][c] && a[r][c]==a[r+1][c] ) return true;
	if(a[r][c]==a[r][c+1] && a[r][c]==a[r][c+2] ) return true;
	if(a[r][c]==a[r][c-1] && a[r][c]==a[r][c-2] ) return true;
	if(a[r][c]==a[r][c+1] && a[r][c]==a[r][c-1] ) return true;
	return false;
}
int main()
{
	int n,m;
	cin>>n>>m;
	for(int i=2;i<2+n;++i)
		for(int j=2;j<2+m;++j)
			cin>>a[i][j];

	for(int i=2;i<2+n;++i){
		for(int j=2;j<2+m;++j){
			if(vaild(i,j)) cout<<0<<' ';
			else cout<<a[i][j]<<' ';
		}
		cout<<endl;
	}
	return 0;
}

// 201512-3 画图

#include <bits/stdc++.h>
using namespace std;
const int N=110;

char g[N][N];
int n,m;
int dire[]={-1,0,1,0,-1};

void draw(int x1,int y1,int x2,int y2){
 if(x1>x2) swap(x1,x2);
 if(y1>y2) swap(y1,y2);
 if(x1==x2){     // 横线
     for(int j=y1;j<=y2;++j){    // !!!!!!! 多条直线相交于一点g[x1][j]=='+'
         g[x1][j]=(g[x1][j]=='|' || g[x1][j]=='+')?'+':'-';
     }
 }else{
     for(int j=x1;j<=x2;++j)            // g[j][y1]=='+'
         g[j][y1]=(g[j][y1]=='-' || g[j][y1]=='+')?'+':'|';
 }
}
void fill(int x,int y,char c){
 if(x<0 || y<0 || x>=n || y>=m) return;
 if(g[x][y]==c || g[x][y]=='-' || g[x][y]=='+' || g[x][y]=='|') return;
 g[x][y]=c;
 

 for(int i=0;i<4;++i){
     int xx=x+dire[i],yy=y+dire[i+1];
     fill(xx,yy,c);
 }
}
int main()
{
 int q;
 cin>>m>>n>>q;
 
 for(int i=0;i<n;++i)
     for(int j=0;j<m;++j)
         g[i][j]='.';

 for(int i=0;i<q;++i){
     int t;
     cin>>t;
     if(t==0){
         int x1,x2,y1,y2;
         cin>>y1>>x1>>y2>>x2;
         draw(x1,y1,x2,y2);
     }else {
         int x,y;
         char c;
         cin>>y>>x>>c;
         fill(x,y,c);
     }
 
 }
 for(int i=n-1;i>=0;--i){
     for(int j=0;j<m;++j)
         cout<<g[i][j];
     cout<<endl;
 }

 
 return 0;
}

//  201512-4 送货
// 题意：n个顶点，m条边，构成的无向图，从编号为1的节点出发，要求每条边只能经过一次，问能否经过所有的边，若能，按字典序输出节点编号
// 思路：
#include <bits/stdc++.h>
using namespace std;

const int N=10010;
vector<int> g[N];
bool vis[N][N];
int n,m;
vector<int> res;
stack<int> st;


int main()
{
    cin>>n>>m;
    for(int i=0;i<m;++i){
        int x,y;
        cin>>x>>y;
        g[x].push_back(y);
        g[y].push_back(x);
    }
    for(int i=1;i<=n;++i)       // ！！！！！！ 先对邻居排个序
        sort(g[i].begin(),g[i].end());

    st.push(1);
    // 以下代码模拟了，dfs一路递归不回溯的过程，就是看能否一路到底遍历完所有的边
    while(!st.empty()){                                                         // ！！！！！！！
        int u=st.top(),i;       // 先别出栈，
        for(i=0;i<g[u].size();++i){
            int v=g[u][i];
            if(vis[u][v]) continue;
            st.push(v);
            vis[v][u]=vis[u][v]=true;       // 该无向边不再使用
            break;      // 入栈到底
        }
        if(i>=g[u].size()){
            st.pop();
            res.push_back(u);       // 所有相临节点遍历完，再出栈，同时加入答案序列
        }
    }
    int cnt=0;
    for(int i=1;i<=n;++i)
        if(g[i].size()%2==1)
            ++cnt;
    // 欧拉图一定是连通图，且无向图的所有结点的出入度均为偶数，或者有2个出入度为奇数的结点。      // ！！！！！！！！
    //  若从结点i出发，如果有2个出入度为奇数的结点，i的出入度必须为奇数。
    if((cnt!=0 && cnt!=2) || (cnt==2 && g[1].size()%2==0) || res.size()!=m+1)
        cout<<-1;
    else
        for(int i=res.size()-1;i>=0;--i) cout<<res[i]<<' ';

    return 0;
}


//201509-1
#include <bits/stdc++.h>
using namespace std;


int main()
{
	int n;
	cin>>n;
	int res=0,p=-1;
	while(n--){
		int a;
		cin>>a;
		if(a!=p) ++res;
		p=a;
	}
	cout<<res<<endl;
	return 0;
}



// 201509-2
#include <iostream>
using namespace std;

bool isr(int n){
	return (n%4==0 && n%100!=0 )|| n%400==0;
}
int mouths[13]={0,31,28,31,30,31,30,31,31,30,31,30,31};
int y,d;
int main()
{
	cin>>y>>d;
	int x=0;
	int i=1;
	if(isr(y)) mouths[2]+=1;
	while(d>0){
		x=d%(mouths[i]+1);
		d-=mouths[i++];
	}
	cout<<i-1<<endl;
	cout<<x<<endl;
	
	return 0;
}



// 201509-3  模板生成系统

//流程，将所有行拼接为一个字符串res，再读入所有的变量和值建立哈希表
//遍历res，寻找{{ 和 }}进行字符串替换
#include <bits/stdc++.h>
using namespace std;

unordered_map<string,string> hs;
void replace(string& str){
 size_t pos=0,next_pos;
 // ！！！！！！！！！！！
 while((pos = str.find("{{ ", pos)) != string::npos)
   {
     next_pos=str.find(" }}",pos);
     string from=str.substr(pos,next_pos+3-pos);
       str.replace(pos, from.length(), hs[from]);
       pos += hs[from].length();
   }
}
int main()
{
 int n,m;
 cin>>n>>m;
 string res,s;
 getchar();
 for(int i=0;i<n;++i){
     getline(cin,s);
     res+=s;
     res+='\n';
 }
 res.pop_back();
 for(int i=0;i<m;++i){
     string t,p;
     cin>>s;
     s="{{ "+s+" }}";        // !!!!!!!!!!!11 。{{ VAR }}
     getchar();
     getline(cin,t);
     
     for(int i=1;i<t.size()-1;++i)
         p+=t[i];
     hs[s]=p;
 }
 replace(res);
 cout<<res<<endl;
 return 0;
 
}

// 201509-4 高速公路
// 题意：求相互连通的点对的个数
// 思路：使用tarjan求连通分量
// tarjan算法求连通分量
#include <bits/stdc++.h>
using namespace std;
const int N=10010,M=100010;
vector<int> he[N];
// dfn[i]为时间戳，low[i]为按照方向进行访问，能够访问到的最早的节点的时间戳
int ins[N],dfn[N],low[N];      // ins[i]记录节点i是否在栈中
int n,m,tot,res,num;
stack<int> stk;
// cnt为连通分量个数
void tarjan(int x){
    dfn[x]=low[x]=++num;        // 时间戳
    stk.push(x);
    ins[x]=1;       // 进栈
    for(auto y:he[x]){
        if(!dfn[y]){        // 没时间说明没访问过
            tarjan(y);      // 递归
            low[x]=min(low[x],low[y]);      // 回溯 ！！！！！！！
        }else if(ins[y]){       // 访问过，但在栈中
            low[x]=min(low[x],dfn[y]);      // ！！！！！！！
        }
    }
    if(dfn[x]==low[x]){
        int y=-1,sum=0;
        while(x!=y){
            y=stk.top();stk.pop();
            ins[y]=0;           // !!!!!!!!!!! 出栈的元素，在栈中的标志要清除
            ++sum;
        }
        res+=sum*(sum-1)/2;
    }
}


int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin>>n>>m;
    for(int i=0;i<m;++i){
        int x,y;
        cin>>x>>y;
        he[x].push_back(y);
    }
    
    for(int i=1;i<=n;++i){          // !!!!!
        if(!dfn[i]) tarjan(i);
    }
    
    cout<<res<<endl;
    return 0;
}



//201503-1


#include <bits/stdc++.h>
using namespace std;

int n,m;
int a[1010][1010];
int main()
{
	cin>>n>>m;
	for(int i=0;i<n;++i)
		for(int j=0;j<m;++j)
			cin>>a[i][j];
	for(int j=m-1;j>=0;--j){
		for(int i=0;i<n;++i)
			cout<<a[i][j]<<' ';
		cout<<endl;
	}
	return 0;
}


//201503-2

#include <bits/stdc++.h>
using namespace std;
pair<int,int> a[1010];
int main()
{
	int n;
	cin>>n;
	for(int i=1;i<=1010;++i)        //注意不是<=n
		a[i].first=i;
	for(int i=0;i<n;++i){
		int t;
       cin>>t;
		a[t].second++;
	}
	sort(a,a+1010,[](const pair<int,int>&a,const pair<int,int>&b){
		if(a.second!=b.second) return a.second>b.second;
		return a.first<b.first;
	});
	
	for(int i=0;i<1010;++i){        // 注意不是<=n
		if(a[i].second==0) break;
		cout<<a[i].first<<' '<<a[i].second<<endl;
	}
	
	
	return 0;
}




//  201503-3    节日

#include <bits/stdc++.h>
using namespace std;

int a,b,c,y1,y2;
int cnt[2051];      // 年的累加和
int month[]={0,31,28,31,30,31,30,31,31,30,31,30,31};
int sm[13];     // 月的累加和

bool isrun(int i){
 if(i%400==0 || (i%4==0 && i%100)) return true;
 return false;
}
int check(int y){
 int k=(2+(cnt[y-1])%7)%7;     // y年的1.1是星期几
 int sum=sm[a-1];
 if(a>=3 && isrun(y)) sum++;
 k=(k+sum%7)%7;              // y年a月的1号是星期几

 int res=1;      // 1号
 while(k!=c){
     ++res;
     k=(k+1)%7;
 }
 res+=7*(b-1);
 return res;
}

int main()
{
 cin>>a>>b>>c>>y1>>y2;
 if(c==7) c=0;
 for(int i=1850;i<=2050;++i){
     cnt[i]=isrun(i)?366:365;
     cnt[i]+=cnt[i-1];
 }
 for(int i=1;i<=12;++i) sm[i]=sm[i-1]+month[i];

 for(int i=y1;i<=y2;++i){
     int d=check(i);
     if(d>month[a]){     // break是提出循环的，不是跳出if的
         if(a==2 && isrun(i) && d==29)
             printf("%d/%02d/%02d\n",i,a,d);
         else
             cout<<"none"<<endl;
         continue;
     }
     printf("%d/%02d/%02d\n",i,a,d);
 }
 return 0;
}


// 201503-4 网络延时
// 正解：此题本质是求树的直径问题，即在树中找两个结点使得它们的距离最大。
// 做法：任意选择一个结点s1，使用bfs或者dfs找到距离结点s1最远的结点s2，
//      然后再对结点s2进行一次bfs或者dfs求距离结点s2最远的结点s3的距离，这个距离就是树的直径。
//      这就找到了最远的两个点s2,s3

#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> PII;

const int N=20010;
int n,m;
vector<int> a[N];
bool vis[N];
// 返回最远的距离和最远距离的编号
PII dfs(int i){
    PII res={0,i};
    for(int j=0;j<a[i].size();++j){
        if(vis[a[i][j]]) continue;      // !!!别忘了标记为 走过
        vis[a[i][j]]=true;
        res=max(res,dfs(a[i][j]));
        vis[a[i][j]]=false;
    }
    res.first++;        // ！！！！！！！！！
    return res;
}
int main()
{
    cin>>n>>m;
    for(int i=2;i<=n+m;++i){
        int j;
        cin>>j;
        a[j].push_back(i);
        a[i].push_back(j);
    }
    int t=dfs(1).second;
    cout<<dfs(t).first-1;       // ！！！！！！！！
}




// 201412-1
#include <bits/stdc++.h>
using namespace std;

const int N=1010;
int cnt[N];
int main()
{
	int n;
	cin>>n;
	for(int i=0;i<n;++i){
		int t;
		cin>>t;
		cout<<++cnt[t]<<' ';
	}
	return 0;
}


//201412-2

//模拟，注意，每次转折完都是沿对角线对称走，且转折遇到边界会改变转折策略
#include <bits/stdc++.h>
using namespace std;

int n;
const int N=510;
int a[N][N];
void draw_z_down(int r,int c){
	int rr=c,cc=r;
	while(r!=rr || c!=cc){
		cout<<a[r][c]<<' ';
		r++;
		c--;
	}
	cout<<a[r][c]<<' ';
}
void draw_z_up(int r,int c){
	int rr=c,cc=r;
	while(r!=rr || c!=cc){
		cout<<a[r][c]<<' ';
		r--;
		c++;
	}
	cout<<a[r][c]<<' ';
}
int main()
{
	cin>>n;
	for(int i=1;i<=n;++i)
		for(int j=1;j<=n;++j)
			cin>>a[i][j];
	cout<<a[1][1]<<' ';
	
	if(n==1) return 0;      // 别忘了n等于1的情况
	
	int i=1,j=2;
	bool turn_down=true;
	while(i!=n || j!=n){
		if(turn_down){
			draw_z_down(i,j);
			swap(i,j);
			if(i<n)
				++i;
			else
				++j;
		}else{
			draw_z_up(i,j);
			swap(i,j);
			if(j<n)
				++j;
			else
				++i;
		}
		turn_down = !turn_down;
	}
	cout<<a[n][n];
	return 0;
}

// 201412-3 集合竞价

#include <bits/stdc++.h>
using namespace std;
const int N=5010;
typedef long long LL;
typedef pair<float,LL> PFLL;

int type[N],s[N];
float p[N];

PFLL buy[N],sell[N];
int nb=0,ns=0;

LL calu(int i){
 LL sum_b=buy[nb].second-buy[i-1].second;
 PFLL t={buy[i].first,LLONG_MAX};        // ！！LLONG_MAX
 int p=upper_bound(sell+1,sell+1+ns,t)-sell;
 LL sum_s=sell[p-1].second;          // ！！！p-1

 return min(sum_s,sum_b);
 
}
int main()
{
 int i=0;
 string line;
 while(getline(cin,line)){
     ++i;
     istringstream iss(line);
     string t;
     
     iss>>t;
     if(t=="cancel") {
         int id;
         iss>>id;
         type[id]=0;
         continue;
     }
     if(t=="buy") type[i]=1;
     else type[i]=2;
     
     iss>>p[i]>>s[i];
 }
 int n=i;
 for(int k=1;k<=n;++k){
     if(type[k]==0) continue;
     if(type[k]==1){
         buy[++nb]={p[k],s[k]};
     }else{
         sell[++ns]={p[k],s[k]};
     }
 }
 

 sort(buy+1,buy+1+nb);
 sort(sell+1,sell+1+ns);

 // 前缀和
 for(int i=2;i<=nb;++i)
     buy[i].second+=buy[i-1].second;

 for(int i=2;i<=ns;++i)
     sell[i].second+=sell[i-1].second;

 LL mxs=LLONG_MIN;
 float mxp=-1;
 // 首先，开票价一定在题目的出价中，因为若取在某个卖价和买价之间，那么自然可以取为更高的买价
 // 其次“系统可以将所有出价至少为p0的买单和所有出价至多为p0的卖单进行匹配”
 // 即高买低卖，从上面这句话可以看出开盘价一定在买单价中取得
 for(int i=1;i<=nb;++i){
     LL res=calu(i);
     if(res>=mxs){
         mxs=res;
         mxp=buy[i].first;
     }
 }
 printf("%.2f %lld",mxp,mxs);

 return 0;
}


// 201412-4 最优灌溉
// 思路：存储边，按边的权值排序，使用并查集检查边的两端的节点是否是在一个集合，若不是，则合并，结果加上该边的权值
#include <bits/stdc++.h>
using namespace std;

typedef pair<int,int> PII;
const int N=1000010;
int n,m;
struct node{
    int x,y,w;
    bool operator<(const node& a) const {
        return w<a.w;
    }
}a[N];
int fa[1010];
int find(int x){
    if(fa[x]!=x) fa[x]=find(fa[x]);
    return fa[x];
}

void Kruskal(){
    sort(a+1,a+1+m);        // 这个排个序，就不用优先队列了
    int res=0;
    for(int i=1;i<=m;++i){
        int x=find(a[i].x),y=find(a[i].y);
        if(x==y) continue;
        res+=a[i].w;
        fa[y]=x;
    }
    cout<<res<<endl;
}
int main()
{
    cin>>n>>m;
    for(int i=1;i<=n;++i) fa[i]=i;
    for(int i=1;i<=m;++i){
        cin>>a[i].x>>a[i].y>>a[i].w;        // 只存边就行
    }

    Kruskal();
    return 0;
}


//201409-1

#include <bits/stdc++.h>
using namespace std;

int main()
{
	int n;
	cin>>n;
	
	vector<int> a;
	for(int i=0;i<n;++i){
		int t;
		cin>>t;
		a.push_back(t);
	}
	sort(a.begin(),a.end());
	int res=0;
	for(int i=0;i<n-1;++i){
		if(a[i]==a[i+1]-1) ++res;
	}
	cout<<res<<endl;
	return 0;
}

//201409-2

//暴力 1e6

#include <bits/stdc++.h>
using namespace std;
const int N=110;
int a[N][N];
int n;
int main()
{
	cin>>n;
   while(n--){
   	int x1,y1,x2,y2;
   	cin>>x1>>y1>>x2>>y2;
		for(int i=x1;i<x2;++i)
			for(int j=y1;j<y2;++j)
				++a[i][j];
	}
	int res=0;
	for(int i=0;i<=100;++i)
		for(int j=0;j<100;++j)
			res+=(a[i][j]>0);
	cout<<res<<endl;
	return 0;
}

// 201409-3 字符串匹配

#include <bits/stdc++.h>
using namespace std;


int main()
{
 string s;
 cin>>s;
 
 int flag;
 cin>>flag;
 
 int n;
 cin>>n;
 for(int i=0;i<n;++i){
     string t;
     cin>>t;
     bool ok=false;
     for(int x=0;x<t.size()-s.size()+1;++x){
         int y;
         for(y=0;y<s.size();++y){
             if(flag){
                 if(s[y]!=t[x+y]) break;
             }else{
                 if(tolower(s[y])!=tolower(t[x+y])) break;
             }
         }
         if(y==s.size()){
             ok=true;
             break;
         }
         
     }
     if(ok) cout<<t<<endl;
 }

 return 0;
}



// 201409-4 最优配餐
// 题意：n*n方格上，有m个分店，k个客户，d个不能经过的点，每个客户可能订多份餐,每份餐一步消耗费用1，求送完所有餐的最小消费
// 思路：所有分店同步bfs，记录客户数量，当客户数量为0,退出，注意同一个位置可能有多个客户
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> PII;

int dire[5]={-1,0,1,0,-1};
const int N=1010;
int a[N][N];        // bfs遍历过的点和不能经过为-1，客户点为正数
int client[N][N];       // 客户数需要单独记下！！！！！
int n,m,k,d;
queue<PII> q;


bool vaild(int x,int y){
    return !(x<=0 || y<=0 || x>n || y>n || a[x][y]<0);      // a[x][y]<0!!!!!
}
long long bfs(){
    int rest=k,len=0;       // ！！！ rest为剩余送餐的客户数量
    long long res=0;        // ！！！！！！！，res要爆int,故要long long,返回值也要
    while(q.size()){
        int c=q.size();
        ++len;
        
        for(int i=0;i<c;++i){
            int x=q.front().first,y=q.front().second;
            q.pop();
            for(int i=0;i<4;++i){
                int xx=x+dire[i],yy=y+dire[i+1];
                if(!vaild(xx,yy)) continue;
                res+=a[xx][yy]*len;
                rest-=client[xx][yy];
                a[xx][yy]=-1;       // ！！！！ 
                q.push({xx,yy});
                
            }
            if(rest==0) break;
        }
        if(rest==0) break;
    }
    return res;
    
}
int main()
{
    cin>>n>>m>>k>>d;
    
    for(int i=0;i<m;++i){
        int x,y;
        cin>>x>>y;
        a[x][y]=-1;     // 分店
        q.push({x,y});  // 同步bfs
    }
    for(int i=0;i<k;++i){
        int x,y,c;
        cin>>x>>y>>c;       // 客户
        a[x][y]+=c;
        client[x][y]++;
    }
    for(int i=0;i<d;++i){
        int x,y;
        cin>>x>>y;      // 不能过的点
        a[x][y]=-1;
    }
    cout<<bfs()<<endl;
    // 所有分店同步bfs
    return 0;
}




// 201403-1
#include <iostream>
using namespace std;

int a[1010];
int main()
{
	int n;
	cin>>n;
	for(int i=0;i<1010;++i) a[i]=1;
	while(n--){
		int t;
		cin>>t;
		if(t>=0) a[t]+=1;
		else a[-t]-=2;
	}
	int res=0;
	for(int i=0;i<1010;++i)
		res+=a[i]==0;
	cout<<res<<endl;
	return 0;
}



// 201403-2
// 暴力，模拟
#include <bits/stdc++.h>
using namespace std;
struct window{
	int id;
	int x1,y1,x2,y2;
};

window a[15];
int main()
{
	int n,m;
	cin>>n>>m;
	for(int i=0;i<n;++i){
		a[i].id=i+1;
		cin>>a[i].x1>>a[i].y1>>a[i].x2>>a[i].y2;
	}
	while(m--){
		int x,y;
		cin>>x>>y;
       int i=n-1;
       for(;i>=0;--i){
			if(x>=a[i].x1 && x<=a[i].x2 && y>=a[i].y1 && y<=a[i].y2) break;
		}
		if(i<0){
			cout<<"IGNORED"<<endl;
		}else{
			cout<<a[i].id<<endl;
			window tmp=a[i];
			for(;i<n-1;++i)
				a[i]=a[i+1];
			a[n-1]=tmp;
		}
	}

	return 0;
}

// 201403-3 命令行选项

#include <bits/stdc++.h>
using namespace std;

int st[26];     // st[1]表无参，st[2]表有参
int main()
{
        // ab:m:" 表示该程序接受三种选项,即"-a"(不带参数),"-b"(带参数), 以及"-m"(带参数)。
     string s;
     cin>>s;
     for(int i=0;i<s.size();++i){
         if(i+1<s.size() && s[i+1]==':') st[s[i++]-'a']=2;
         else st[s[i]-'a']=1;
     }

     int n;
     cin>>n;
     getchar();      // ！！！！
     for(int i=1;i<=n;++i){
     
         getline(cin,s);
         stringstream ss(s);
         string res[26]={};
         
         char p=0;
         getline(ss,s,' ');
         while(getline(ss,s,' ')){

             if(s[0]=='-' && islower(s[1])){     // 选项
                 if(st[s[1]-'a']==0) break;     // 选项错误
                 if(st[s[1]-'a']==1) res[s[1]-'a']="N";      // 由于不会出现大写字母，使用N表无参
                 if(st[s[1]-'a']==2) {       // 有参
                     string t;
                     if(getline(ss,t,' ')==0) break;     // 本有参数，但无参数时，错误  ！！！！！！！！！！！！！！！
                     res[s[1]-'a']=t;
                 }
             }else
                 break;


         }
         cout<<"Case "<<i<<':';
         for(int i=0;i<26;++i){
             if(res[i].empty()) continue;
             cout<<" -"<<(char)('a'+i);
             if(res[i]!="N") cout<<" "<<res[i];
         }
         cout<<endl;
     }
     
     return 0;
}


// 201403-4
// 题意：平面上有n个路由器，只要两个路由间距不超过r就能连接，要使得路由器1和路由器2相连接，可以在指定的m个位置增加k个路由器，求经过最少路由器个数
// 思路：直接将m个位置都放上路由器，直接bfs找最短路
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef pair<int,int> PII;

const int N = 210;
int dist[N][N];     //dist[i][j]表示 第i个点的路径中增设路由器的数量为j的最短距离。!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bool vis[N][N];
int x[N],y[N];
int n,m,k,r;
bool vaild(int i,int j){
    return 1LL*(x[i]-x[j])*(x[i]-x[j])+1LL*(y[i]-y[j])*(y[i]-y[j]) <= 1LL*r*r;      // ！！！！！！！！ 1LL
}
void bfs(){
    memset(dist,0x3f,sizeof dist);
    dist[1][0]=0;
    queue<PII> q;
    q.push({1,0});      // 新增的个数信息也要考虑
    vis[1][0]=true;     

    while(q.size()){
        auto tmp=q.front();
        q.pop();
        int t=tmp.first,cnt=tmp.second;
        int limit=cnt<k?n+m:n;

        for(int i=1;i<=limit;++i){
            if(!vaild(i,t)) continue;
            if(i<=n){
                if(vis[i][cnt]) continue;
                vis[i][cnt]=true;
                q.push({i,cnt});
                dist[i][cnt]=dist[t][cnt]+1;
            }else{
                if(vis[i][cnt+1]) continue;
                vis[i][cnt+1]=true;
                q.push({i,cnt+1});
                dist[i][cnt+1]=dist[t][cnt]+1;
            }
        }
    }
}
int main()
{
    cin>>n>>m>>k>>r;
    for(int i=1;i<=n+m;++i){
        cin>>x[i]>>y[i];
    }

    bfs();
    int res=INT_MAX;
    for(int i=0;i<=k;++i)
        res=min(res,dist[2][i]);
    cout<<res-1<<endl;

    return 0;
}



//201312-1

#include <bits/stdc++.h>
using namespace std;
int cnt[10010];
int main()
{
	int n;
	cin>>n;
	for(int i=0;i<n;++i){
		int t;
		cin>>t;
		++cnt[t];
	}
	int res,mx=INT_MIN;
	for(int i=1;i<=10000;++i){
		if(cnt[i]>mx){
			mx=cnt[i];
			res=i;
		}
	}
	cout<<res<<endl;
	return 0;
}

// 201312-2
#include <bits/stdc++.h>
using namespace std;

int main()
{
	string s;
	cin>>s;
	int sum=0;
	int cnt=0;
	for(int i=0;i<(int)s.size()-1;++i){
		if(isdigit(s[i])){
			sum+=(s[i]-'0')*(++cnt);
		}
	}
	sum%=11;
	if(sum==10) sum='X';
	else sum+='0';

	if(sum==s.back()) cout<<"Right"<<endl;
	else{
		s.back()=sum;
		cout<<s<<endl;
	}
	return 0;
}

// 201312-3

// 暴力 n*n
#include <bits/stdc++.h>
using namespace std;
int n;
const int N=1010;
int a[N];

int main()
{
 cin>>n;
 for(int i=1;i<=n;++i)
     cin>>a[i];
 
 int res=-1;
 for(int i=1;i<=n;++i){
     int mi=INT_MAX;
     for(int j=i;j<=n;++j){
         mi=min(mi,a[j]);
         res=max((j-i+1)*mi,res);
     }
 }
 cout<<res<<endl;
 return 0;
}


// 201312-4 有趣的数
#include <bits/stdc++.h>
using namespace std;

typedef long long LL;
const int N=1010,MOD=1e9+7;
int n;
int C[N][N];
int main()
{
    cin>>n;
    for(int i=0;i<=n;++i)
        for(int j=0;j<=i;++j)
            if(!j) C[i][j]=1;
            else C[i][j]=(C[i-1][j]+C[i-1][j-1])%MOD;   // 组合的递推公式

    int res=0;
    for(int k=2;k<=n-2;++k)     // k为01的个数
        res=(res+(LL)C[n-1][k]*(k-1) %MOD *(n-k-1))%MOD;        
    // 首位只能放2，故从剩余的n-1位中选k个位置放01，而0都在1前，0固定了1也就固定了，0的个数最多k-1个           ！！！！！！！！！
    // 同理23也是一样，23共n-k个，2固定了3就固定了，2最多n-k-1个     ！！！！！！！！！！！！

    cout<<res;
        
    return 0;
}