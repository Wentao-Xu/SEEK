#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <random>
#include <omp.h>
#include <cassert>
#include <cstring>
#include <cctype>
#include <iterator>

using namespace std;

static default_random_engine GLOBAL_GENERATOR;
static uniform_real_distribution<double> UNIFORM(0, 1);

typedef tuple<int, int, int> triplet;

vector<string> ReadFirstColumn(const string& fname) {
    ifstream ifs(fname, ios::in);

    string line;
    string item;
    vector<string> items;

    assert(!ifs.fail());

    while (getline(ifs, line)) {
        stringstream ss(line);
        ss >> item;
        items.push_back(item);
    }
    ifs.close();

    return items;
}

unordered_map<string, int> CreateIdMapping(const vector<string>& items) {
    unordered_map<string, int> map;
    for (int i = 0; i < (int) items.size(); i++)
        map[items[i]] = i;

    return map;
}

vector<triplet> Create_hrts(
        const string& fname,
        const unordered_map<string, int>& ent_map,
        const unordered_map<string, int>& rel_map) {

    ifstream ifs(fname, ios::in);

    string line;
    string h, r, t;
    vector<triplet> hrts;

    assert(!ifs.fail());

    while (getline(ifs, line)) {
        stringstream ss(line);
        ss >> h >> r >> t;
        hrts.push_back( make_tuple(ent_map.at(h), rel_map.at(r), ent_map.at(t)) );
    }
    ifs.close();

    return hrts;
}

int64_t Encode(int r, int po) {
    return (r << 20) + po;
}

unordered_map<int64_t, unordered_set<int>> FindRelatedEntities(vector<triplet> hrts_tr) {
	unordered_map<int64_t, unordered_set<int>> res;
	for (auto hrt : hrts_tr) {
		int h = get<0>(hrt);
		int r = get<1>(hrt);
		int t = get<2>(hrt);

		auto rhp = Encode(r, 1);
		auto rtp = Encode(r, 2);
		if (res.count(rhp) > 0) {
			res[rhp].insert(h);
		}
		else {
			res[rhp] = {};
			res[rhp].insert(h);
		}
		if (res.count(rtp) > 0) {
			res[rtp].insert(t);
		}
		else {
			res[rtp] = {};
			res[rtp].insert(t);
		}
	}
	return res;
}

vector<vector<double>> UniformMatrix(int m, int n, double l, double h) {
    vector<vector<double>> matrix;
    matrix.resize(m);
    for (int i = 0; i < m; i++)
        matrix[i].resize(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;

    return matrix;
}

vector<vector<double>> ConstMatrix(int m, int n, double c) {
    vector<vector<double>> matrix;
    matrix.resize(m);
    for (int i = 0; i < m; i++)
        matrix[i].resize(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = c;

    return matrix;
}

string StringToUpper(string s){
	string tmp = s;
	for(unsigned i = 0; i < tmp.size(); i++){
		tmp[i] = toupper(tmp[i]);
	}
	return tmp;
}

vector<int> Range(int n) {  // 0 ... n-1
    vector<int> v;
    v.reserve(n);
    for (int i = 0; i < n; i++)
        v.push_back(i);
    return v;
}

double Sigmoid(double x, double cutoff=30) {
    if (x > +cutoff) return 1.;
    if (x < -cutoff) return 0.;
    return 1./(1.+exp(-x));
}

class HRTBucket {
	unordered_set<int64_t> __hrts;
	unordered_map<int64_t, vector<int>> __hr2t;
	unordered_map<int64_t, vector<int>> __tr2h;
	unordered_map<int64_t, unordered_set<int>> __hr2t_s;
	unordered_map<int64_t, unordered_set<int>> __tr2h_s;

	int64_t hash(int a, int b, int c) const {
		int64_t x = a;
		x = (x << 20) + b;
		return (x << 20) + c;
	}

	int64_t hash(int a, int b) const {
		int64_t x = a;
		return (x << 32) + b;
	}

public:
	HRTBucket() {}
	HRTBucket(const vector<triplet>& hrts) {
		for (auto hrt : hrts) {
			int h = get<0>(hrt);
			int r = get<1>(hrt);
			int t = get<2>(hrt);

			int64_t __hrt = hash(h, r, t);
			__hrts.insert(__hrt);

			int64_t __sr = hash(h, r);
			if (__hr2t.find(__sr) == __hr2t.end())
				{
					__hr2t[__sr] = vector<int>();
					__hr2t_s[__sr] = unordered_set<int>();
				}
			if(__hr2t_s[__sr].count(t) <= 0){	
				__hr2t[__sr].push_back(t);
				__hr2t_s[__sr].insert(t);
			}

			int64_t __or = hash(t, r);
			if (__tr2h.find(__or) == __tr2h.end())
				{
					__tr2h[__or] = vector<int>();
					__tr2h_s[__or] = unordered_set<int>();
				}
			if(__tr2h_s[__or].count(h) <=0){
				__tr2h[__or].push_back(h);
				__tr2h_s[__or].insert(h);
			}
		}
	}

	bool Contains(int a, int b, int c) const {
		return __hrts.find(hash(a, b, c)) != __hrts.end();
	}

	vector<int> hr2t(int h, int r) const {
		return __hr2t.at(hash(h, r));
	}

	vector<int> tr2h(int t, int r) const {
		return __tr2h.at(hash(t, r));
	}
};

class NegativeSampler {
	uniform_int_distribution<int> unif_e;
	uniform_int_distribution<int> unif_r;
	default_random_engine generator;

public:
	NegativeSampler(int ne, int nr, int seed) :
		unif_e(0, ne - 1), unif_r(0, nr - 1), generator(seed){
	}

	int RandomEntity() {
		return unif_e(generator);
	}

	int RandomRelation() {
		int r = unif_r(generator);
		return r;
	}
};


class NegativeTripletsSampler {
	int ne;
	default_random_engine generator;
	HRTBucket hrt_bucket_tr;
	unordered_map<int64_t, unordered_set<int>> related_entities;
public:
	NegativeTripletsSampler(int ne, int seed, HRTBucket hrt_bucket_tr, unordered_map<int64_t, unordered_set<int>> related_entities) :
	ne(ne), generator(seed), hrt_bucket_tr(hrt_bucket_tr), related_entities(related_entities) {
	}

	int RandomSubject(int r, int t) {
		auto rsp = Encode(r, 1);
		auto tmpset = related_entities[rsp];
		vector<int> ent;
		for (auto it = tmpset.begin(); it != tmpset.end(); it++) {
			ent.push_back(*it);
		}
		int res;
		int count = 0;
		uniform_int_distribution<int> unif_e(0, ent.size() - 1);
		auto hi = unif_e(generator);
		while (hrt_bucket_tr.Contains(ent[hi], r, t)) {
			hi = unif_e(generator);
			count++;
			if (count > 20)
				break;
		}
		res = ent[hi];
		if (count > 20 || ent.size() < 2) {
			uniform_int_distribution<int> unif_ee(0, ne - 1);
			hi = unif_ee(generator);
			while (hrt_bucket_tr.Contains(hi, r, t)) {
				hi = unif_ee(generator);
			}
			res = hi;
		}
		return res;
	}

	int RandomObject(int h, int r) {
		int res = 0;
		auto rop = Encode(r, 2);
		auto tmpset = related_entities[rop];
		vector<int> ent;
		for (auto it = tmpset.begin(); it != tmpset.end(); it++) {
			ent.push_back(*it);
		}
		int count = 0;
		uniform_int_distribution<int> unif_e(0, ent.size() - 1);
		auto ti = unif_e(generator);
		while (hrt_bucket_tr.Contains(h, r, ent[ti])) {
			ti = unif_e(generator);
			count++;
			if (count > 20)
				break;
		}
		res = ent[ti];
		if (count > 20 || ent.size() < 2) {
			uniform_int_distribution<int> unif_ee(0, ne - 1);
			ti = unif_ee(generator);
			while (hrt_bucket_tr.Contains(h, r, ti)) {
				ti = unif_ee(generator);
			}
			res = ti;
		}
		return res;
	}

};




class Model {
protected:
	double lr;
	double lambda;
	const double init_b = 1e-2;
	const double init_e = 1e-6;

	vector<vector<double>> E;
	vector<vector<double>> R;
	vector<vector<double>> E_g;
	vector<vector<double>> R_g;

public:

	Model(double lr, double lambda) {
		this->lr = lr;
		this->lambda = lambda;
	}

	void Save(const string& fname) {
		ofstream ofs(fname, ios::out);

		for (unsigned i = 0; i < E.size(); i++) {
			for (unsigned j = 0; j < E[i].size(); j++)
				ofs << E[i][j] << ' ';
			ofs << endl;
		}

		for (unsigned i = 0; i < R.size(); i++) {
			for (unsigned j = 0; j < R[i].size(); j++)
				ofs << R[i][j] << ' ';
			ofs << endl;
		}

		ofs.close();
	}

	void Load(const string& fname) {
		ifstream ifs(fname, ios::in);
    	assert(!ifs.fail());
		for (unsigned i = 0; i < E.size(); i++)
			for (unsigned j = 0; j < E[i].size(); j++)
				ifs >> E[i][j];

		for (unsigned i = 0; i < R.size(); i++)
			for (unsigned j = 0; j < R[i].size(); j++)
				ifs >> R[i][j];

		ifs.close();
	}

	void AdagradUpdate(
		int h,
		int r,
		int t,
		const vector<double>& d_h,
		const vector<double>& d_r,
		const vector<double>& d_t) {

		for (unsigned i = 0; i < E[h].size(); i++) E_g[h][i] += d_h[i] * d_h[i];
		for (unsigned i = 0; i < R[r].size(); i++) R_g[r][i] += d_r[i] * d_r[i];
		for (unsigned i = 0; i < E[t].size(); i++) E_g[t][i] += d_t[i] * d_t[i];

		for (unsigned i = 0; i < E[h].size(); i++) E[h][i] -= lr * d_h[i] / sqrt(E_g[h][i]);
		for (unsigned i = 0; i < R[r].size(); i++) R[r][i] -= lr * d_r[i] / sqrt(R_g[r][i]);
		for (unsigned i = 0; i < E[t].size(); i++) E[t][i] -= lr * d_t[i] / sqrt(E_g[t][i]);
	}

	void Train(int h, int r, int t, double label) {
		vector<double> d_h;
		vector<double> d_r;
		vector<double> d_t;

		d_h.resize(E[h].size());
		d_r.resize(R[r].size());
		d_t.resize(E[t].size());

		// double offset = label > 0 ? 1 : 0;
		// double d_loss = Sigmoid(Score(h, r, t)) - offset; // 1 0

		double d_loss = -1.0 * label * Sigmoid(-1.0* label * Score(h, r, t)); 
		// double p = Score(h, r, t);
		// double d_loss = -1.0 * label * Sigmoid(-1. * p) + (1. - label) * Sigmoid(p); // 1 0

		ScoreGrad(h, r, t, d_h, d_r, d_t);
		for (unsigned i = 0; i < d_h.size(); i++) d_h[i] *= d_loss;
		for (unsigned i = 0; i < d_r.size(); i++) d_r[i] *= d_loss;
		for (unsigned i = 0; i < d_t.size(); i++) d_t[i] *= d_loss;

		double lambda_h = lambda / d_h.size();
		double lambda_r = lambda / d_r.size();
		double lambda_t = lambda / d_t.size();

		for (unsigned i = 0; i < d_h.size(); i++) d_h[i] += lambda_h * E[h][i];
		for (unsigned i = 0; i < d_r.size(); i++) d_r[i] += lambda_r * R[r][i];
		for (unsigned i = 0; i < d_t.size(); i++) d_t[i] += lambda_t * E[t][i];

		AdagradUpdate(h, r, t, d_h, d_r, d_t);
	}

	virtual double Score(int h, int r, int t) const = 0;

	virtual void ScoreGrad(
		int h,
		int r,
		int t,
		vector<double>& d_h,
		vector<double>& d_r,
		vector<double>& d_t) {};
};

class Evaluator {
	int ne;
	int nr;
	const vector<triplet>& hrts;
	const HRTBucket& hrt_bucket;

public:
	Evaluator(int ne, int nr, const vector<triplet>& hrts, const HRTBucket& hrt_bucket) :
		ne(ne), nr(nr), hrts(hrts), hrt_bucket(hrt_bucket) {}

	unordered_map<string, double> evaluate(const Model *model, int truncate) {
		int N = this->hrts.size();

		if (truncate > 0)
			N = min(N, truncate);

		double mrr_h = 0.;
		double mrr_r = 0.;
		double mrr_t = 0.;

		double mrr_h_raw = 0.;
		double mrr_t_raw = 0.;

		double mr_h = 0.;
		double mr_r = 0.;
		double mr_t = 0.;

		double mr_h_raw = 0.;
		double mr_t_raw = 0.;

		double hits01_h = 0.;
		double hits01_r = 0.;
		double hits01_t = 0.;

		double hits03_h = 0.;
		double hits03_r = 0.;
		double hits03_t = 0.;

		double hits10_h = 0.;
		double hits10_r = 0.;
		double hits10_t = 0.;

#pragma omp parallel for reduction(+: mrr_h, mrr_r, mrr_t, mr_h, mr_r, mr_t, \
        hits01_h, hits01_r, hits01_t, hits03_h, hits03_r, hits03_t, hits10_h, hits10_r, hits10_t)
		for (int i = 0; i < N; i++) {
			auto ranks = this->Rank(model, hrts[i]);

			double rank_h = get<0>(ranks);
			double rank_r = get<1>(ranks);
			double rank_t = get<2>(ranks);
			double rank_h_raw = get<3>(ranks);
			double rank_t_raw = get<4>(ranks);

			mrr_h += 1. / rank_h;
			mrr_r += 1. / rank_r;
			mrr_t += 1. / rank_t;

			mrr_h_raw += 1. / rank_h_raw;
			mrr_t_raw += 1. / rank_t_raw;

			mr_h += rank_h;
			mr_r += rank_r;
			mr_t += rank_t;
			mr_h_raw += rank_h_raw;
			mr_t_raw += rank_t_raw;

			hits01_h += rank_h <= 01;
			hits01_r += rank_r <= 01;
			hits01_t += rank_t <= 01;

			hits03_h += rank_h <= 03;
			hits03_r += rank_r <= 03;
			hits03_t += rank_t <= 03;

			hits10_h += rank_h <= 10;
			hits10_r += rank_r <= 10;
			hits10_t += rank_t <= 10;
		}

		unordered_map<string, double> info;

		info["mrr_h"] = mrr_h / N;
		info["mrr_r"] = mrr_r / N;
		info["mrr_t"] = mrr_t / N;
		info["mrr_h_raw"] = mrr_h_raw / N;
		info["mrr_t_raw"] = mrr_t_raw / N;

		info["mr_h"] = mr_h / N;
		info["mr_r"] = mr_r / N;
		info["mr_t"] = mr_t / N;
		info["mr_h_raw"] = mr_h_raw / N;
		info["mr_t_raw"] = mr_t_raw / N;

		info["hits03_r"] = hits03_r / N;
		info["hits03_t"] = hits03_t / N;
		info["hits01_h"] = hits01_h / N;
		info["hits01_r"] = hits01_r / N;
		info["hits01_t"] = hits01_t / N;

		info["hits03_h"] = hits03_h / N;

		info["hits10_h"] = hits10_h / N;
		info["hits10_r"] = hits10_r / N;
		info["hits10_t"] = hits10_t / N;

		return info;
	}

private:

	tuple<double, double, double, double, double> Rank(const Model *model, const triplet& hrt) {
		int rank_h = 1;
		int rank_r = 1;
		int rank_t = 1;

		int h = get<0>(hrt);
		int r = get<1>(hrt);
		int t = get<2>(hrt);

		// XXX:
		// There might be degenerated cases when all output Scores == 0, leading to perfect but meaningless results.
		// A quick fix is to add a small offset to the base_Score.
		double base_Score = model->Score(h, r, t) - 1e-32;

		for (int hh = 0; hh < ne; hh++)
			if (model->Score(hh, r, t) > base_Score) rank_h++;

		for (int rr = 0; rr < nr; rr++)
			if (model->Score(h, rr, t) > base_Score) rank_r++;

		for (int tt = 0; tt < ne; tt++)
			if (model->Score(h, r, tt) > base_Score) rank_t++;

		int rank_h_raw = rank_h;
		int rank_t_raw = rank_t;


		for (auto hh : hrt_bucket.tr2h(t, r))
			if (model->Score(hh, r, t) > base_Score) rank_h--;
			// }
		for (auto tt : hrt_bucket.hr2t(h, r))
			if (model->Score(h, r, tt) > base_Score) rank_t--;

		return make_tuple(rank_h, rank_r, rank_t, rank_h_raw, rank_t_raw);
	}
};

class Classifier {
	int ne;
	int nr;
	const vector<int>& tlabel;
	const vector<int>& vlabel;
	const vector<triplet>& test_set;
	const vector<triplet>& valid_set;
public:
	Classifier(int ne, int nr, const vector<int>& tlabel, const vector<int>& vlabel, const vector<triplet>& test_set, const vector<triplet>& valid_set) :ne(ne), nr(nr), tlabel(tlabel), vlabel(vlabel), test_set(test_set), valid_set(valid_set) {
	}

	double Classfy(const Model *model) {
		unordered_map<int, double> R_TH;
		int TN = tlabel.size();
		int VN = vlabel.size();

		unordered_map<int, vector<double>> ttScore;
		unordered_map<int, vector<double>> ntScore;
		for (int i = 0; i < VN; i++) {
			int h = get<0>(valid_set[i]);
			int r = get<1>(valid_set[i]);
			int t = get<2>(valid_set[i]);

			auto sScore = model->Score(h, r, t);

			if (vlabel[i] == 1) {
				ttScore[r].push_back(sScore);
			}
			else {
				ntScore[r].push_back(sScore);
			}
		}

		for (int i = 0; i < nr; i++) {
			auto tmpt = ttScore[i];
			auto tmpn = ntScore[i];
			if (tmpt.size() != 0 && tmpn.size() != 0) {
				sort(tmpt.begin(), tmpt.end());
				sort(tmpn.begin(), tmpn.end());
				int ti = 0;
				int ni = 0;
				vector<int> re;
				while (ti < int(tmpt.size())) {
					if (tmpt[ti] > tmpn[ni]) {
						if (ni < int(tmpn.size() - 1)) {
							ni++;
						}
						else {
							re.push_back(0);
							ti++;
						}
					}
					else {
						re.push_back(tmpn.size() - ni);
						ti++;
					}
				}
				int ma = 10000000;
				int index = 0;
				for (int i = 0; i < int(re.size()); i++) {
					if (re[i] + i < ma) {
						ma = re[i] + i;
						index = i;
					}
				}

				R_TH[i] = tmpt[index];
			}
			else {
				R_TH[i] = 0.01;
			}
		}

		double true_triplets = 0.;
		for (int i = 0; i < TN; i++) {
			int h = get<0>(test_set[i]);
			int r = get<1>(test_set[i]);
			int t = get<2>(test_set[i]);
			auto tScore = model->Score(h, r, t);
			int l = 0;
			if (tScore >= R_TH[r])
				l = 1;
			else
				l = 0;
			if (l == tlabel[i])
				true_triplets++;
		}

		return true_triplets / TN;
	}
};

void PrettyPrint(const char* prefix, const unordered_map<string, double>& info) {
    printf("%s  Metrics    \t H \t R \t T\n", prefix);	
    printf("%s  MRR    \t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_h"),    100*info.at("mrr_r"),    100*info.at("mrr_t"));
    printf("%s  MRR_RAW\t%.2f\t\t%.2f\n", prefix, 100*info.at("mrr_h_raw"),    100*info.at("mrr_t_raw"));
    printf("%s  MR     \t%.2f\t%.2f\t%.2f\n", prefix, info.at("mr_h"), info.at("mr_r"), info.at("mr_t"));
    printf("%s  MR_RAW \t%.2f\t\t%.2f\n", prefix, info.at("mr_h_raw"), info.at("mr_t_raw"));
    printf("%s  Hits@01\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits01_h"), 100*info.at("hits01_r"), 100*info.at("hits01_t"));
    printf("%s  Hits@03\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits03_h"), 100*info.at("hits03_r"), 100*info.at("hits03_t"));
    printf("%s  Hits@10\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits10_h"), 100*info.at("hits10_r"), 100*info.at("hits10_t"));
}

void MrrPrint(const char* prefix, const unordered_map<string, double>& info) {
    printf("%s  Metrics    \t H \t R \t T \tEntAverage\n", prefix);	
    printf("%s  MRR    \t%.2f\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_h"),    100*info.at("mrr_r"),    100*info.at("mrr_t"), 100*(info.at("mrr_h") + info.at("mrr_t"))/2.0);
    printf("%s  MRR_RAW\t%.2f\t\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_h_raw"),    100*info.at("mrr_t_raw"), 100*(info.at("mrr_h_raw") + info.at("mrr_t_raw"))/2.0);
}

void HitPrint(const char* prefix, const unordered_map<string, double>& info) {
    printf("%s  Hits@01\t%.2f\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits01_h"), 100*info.at("hits01_r"), 100*info.at("hits01_t"), 100*(info.at("hits01_h") + info.at("hits01_t"))/2.0);
    printf("%s  Hits@03\t%.2f\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits03_h"), 100*info.at("hits03_r"), 100*info.at("hits03_t"), 100*(info.at("hits03_h") + info.at("hits03_t"))/2.0);
    printf("%s  Hits@10\t%.2f\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits10_h"), 100*info.at("hits10_r"), 100*info.at("hits10_t"), 100*(info.at("hits10_h") + info.at("hits10_t"))/2.0);
}
// based on Google's word2vec
int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

class SEEK : public Model { 
	int nh;
	int snh;
	int num_seg;

public:
	SEEK(int ne, int nr, int nh, int num_seg, double lr, double lambda) : Model(lr, lambda) {
		this->nh = nh;
		this->num_seg = num_seg;
		this->snh = nh / num_seg;
		E = UniformMatrix(ne, nh, -init_b, init_b);
		R = UniformMatrix(nr, nh, -init_b, init_b);
		E_g = ConstMatrix(ne, nh, init_e);
		R_g = ConstMatrix(nr, nh, init_e);
	}

	double Score(int h, int r, int t) const {
		double dot = 0;
		for (int k = 0; k < num_seg; k++) {
			int r_i = k * snh;
			for (int j = 0; j < num_seg; j++) {
				int h_i = j * snh;
				int t_i = h_i;
				double sij = 1.0;
				if (k & 1) {
					t_i = ((j + k) % num_seg) *snh;
					if (j > (j + k) % num_seg) {
						sij = -1.0;
					}
				}
				for (int i = 0; i < snh; i++) {
					dot += sij * R[r][r_i + i] * E[h][h_i + i] * E[t][t_i + i];
				}
			}
		}
		return dot;
	}

	void ScoreGrad(
		int h,
		int r,
		int t,
		vector<double>& d_h,
		vector<double>& d_r,
		vector<double>& d_t) {

		for (int k = 0; k < num_seg; k++) {
			int r_i = k * snh;
			for (int j = 0; j < num_seg; j++) {
				int h_i = j * snh;
				int t_i = h_i;
				double sij = 1.0;
				if (k & 1) {
					t_i = ((j + k) % num_seg) *snh;
					if (j > (j + k) % num_seg) {
						sij = -1.0;
					}
				}
				for (int i = 0; i < snh; i++) {
					d_r[r_i + i] += sij * E[h][h_i + i] * E[t][t_i + i];
					d_h[h_i + i] += sij * R[r][r_i + i] * E[t][t_i + i];
					d_t[t_i + i] += sij * R[r][r_i + i] * E[h][h_i + i];
				}
			}
		}
	}
};


int main(int argc, char **argv) {
	// option parser
	//string  dataset = "FB15K";
	string  dataset = "FB15K";
	string  algorithm = "SEEK";
	int embed_dim = 200;
	int  num_seg = 8;
	double  lr = 0.1;
	double  lambda = 1e-3;
	int neg_sample = 10;
	int     num_epoch = 500;
	int     num_thread = 24;
	int     eval_freq = 50;
	string  model_path;
	bool classification = false;
	bool    prediction = false;
	string datasetpath = "FB15k/freebase_mtr100_mte100";
	int i;
	if ((i = ArgPos((char *)"-dataset", argc, argv)) > 0) dataset = StringToUpper(string(argv[i + 1]));
	if(dataset == "FB15K"){
		datasetpath = "FB15k/freebase_mtr100_mte100";
		lambda = 1e-3;
		embed_dim = 400;
		neg_sample = 1000;
		num_epoch = 100;
		eval_freq = 5;
		num_seg = 8;
	}
	else if(dataset == "DB100K"){
		datasetpath = "DB100K/db100k";
		lambda = 1e-2;
		embed_dim = 400;
		neg_sample = 100;
		num_epoch = 100;
		eval_freq = 5;
		num_seg = 4;
	}
   else if(dataset == "YAGO37"){
        datasetpath = "yago37/yago37";
        lambda = 1e-3;
        embed_dim = 400;
        neg_sample = 200;
        num_epoch = 100;
        eval_freq = 1;
        num_seg = 8;
    }

	if ((i = ArgPos((char *)"-algorithm", argc, argv)) > 0)  algorithm = string(argv[i + 1]);
	if ((i = ArgPos((char *)"-embed_dim", argc, argv)) > 0)  embed_dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-lr", argc, argv)) > 0)  lr = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-lambda", argc, argv)) > 0)  lambda = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-neg_sample", argc, argv)) > 0)  neg_sample = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-num_epoch", argc, argv)) > 0)  num_epoch = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-num_thread", argc, argv)) > 0)  num_thread = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-eval_freq", argc, argv)) > 0)  eval_freq = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-model_path", argc, argv)) > 0)  model_path = string(argv[i + 1]);
	if ((i = ArgPos((char *)"-prediction", argc, argv)) > 0)  prediction = true;
	if ((i = ArgPos((char *)"-classification", argc, argv)) > 0)  classification = true;
	if ((i = ArgPos((char *)"-num_seg", argc, argv)) > 0)  num_seg = atoi(argv[i + 1]);	
		
	printf("dataset     =  %s\n", dataset.c_str());
	printf("algorithm   =  %s\n", algorithm.c_str());
	printf("embed_dim   =  %d\n", embed_dim);
	printf("num_seg     =  %d\n", num_seg);
	printf("lr          =  %e\n", lr);
	printf("lambda      =  %e\n", lambda);
	printf("neg_sample  =  %d\n", neg_sample);
	printf("num_epoch   =  %d\n", num_epoch);
	printf("num_thread  =  %d\n", num_thread);
	printf("eval_freq   =  %d\n", eval_freq);
	printf("model_path  =  %s\n", model_path.c_str());

	vector<string> ents = ReadFirstColumn(datasetpath + "-entities.txt");
	vector<string> rels = ReadFirstColumn(datasetpath + "-relations.txt");

	unordered_map<string, int> ent_map = CreateIdMapping(ents);
	unordered_map<string, int> rel_map = CreateIdMapping(rels);

	int ne = ent_map.size();
	int nr = rel_map.size();

	vector<triplet> hrts_tr = Create_hrts(datasetpath + "-train.txt", ent_map, rel_map);
	vector<triplet> hrts_va = Create_hrts(datasetpath + "-valid.txt", ent_map, rel_map);
	vector<triplet> hrts_te = Create_hrts(datasetpath + "-test.txt", ent_map, rel_map);
	vector<triplet> hrts_al;

	hrts_al.insert(hrts_al.end(), hrts_tr.begin(), hrts_tr.end());
	hrts_al.insert(hrts_al.end(), hrts_va.begin(), hrts_va.end());
	hrts_al.insert(hrts_al.end(), hrts_te.begin(), hrts_te.end());

	HRTBucket hrt_bucket_al(hrts_al);
	HRTBucket hrt_bucket_tr(hrts_tr);

	Model *model = new SEEK(ne, nr, embed_dim, num_seg, lr, lambda);

	string model_path_mrr = "Mrr" + model_path;
	string model_path_hit = "Hit" + model_path;
	
	if (prediction) {
		cout<< ne<< " " << nr <<endl;
		Evaluator evaluator_te(ne, nr, hrts_te, hrt_bucket_al);
		model->Load(model_path_mrr);
		auto info_te_mrr = evaluator_te.evaluate(model, -1);
		model->Load(model_path_hit);
		auto info_te_hit = evaluator_te.evaluate(model, -1);
		MrrPrint("TE", info_te_mrr);
		HitPrint("TE", info_te_hit);
		return 0;
	}

	if (classification) {
		auto related_entities = FindRelatedEntities(hrts_al);
		model->Load(model_path_mrr);
		NegativeTripletsSampler sample(ne, rand(), hrt_bucket_tr, related_entities);
		vector<int> tlabel;
		vector<int> vlabel;
		vector<triplet> test_set;
		vector<triplet> valid_set;

		for (auto hrt : hrts_te) {
			int h = get<0>(hrt);
			int r = get<1>(hrt);
			int t = get<2>(hrt);
			test_set.push_back(hrt);
			tlabel.push_back(1);

			int c = 0;
			if (related_entities[Encode(r, 1)].size() > related_entities[Encode(r, 2)].size())
				c = 1;
			else
				c = 0;
			if (c == 0)
				test_set.push_back(make_tuple(sample.RandomSubject(r, t), r, t));
			else
				test_set.push_back(make_tuple(h, r, sample.RandomObject(h, r)));
			tlabel.push_back(0);
		}

		for (auto hrt : hrts_va) {
			int h = get<0>(hrt);
			int r = get<1>(hrt);
			int t = get<2>(hrt);
			valid_set.push_back(hrt);
			vlabel.push_back(1);

			int c = 0;
			if (related_entities[Encode(r, 1)].size() > related_entities[Encode(r, 2)].size())
				c = 1;
			else
				c = 0;
			if (c == 0)
				valid_set.push_back(make_tuple(sample.RandomSubject(r, t), r, t));
			else
				valid_set.push_back(make_tuple(h, r, sample.RandomObject(h, r)));
			vlabel.push_back(0);
		}

		Classifier classifier(ne, nr, tlabel, vlabel, test_set, valid_set);
		auto accuracy = classifier.Classfy(model);
		cout << "accuracy of triplets classification : " << accuracy << endl;
		return 0;
	}

	Evaluator evaluator_va(ne, nr, hrts_va, hrt_bucket_al);
	Evaluator evaluator_tr(ne, nr, hrts_tr, hrt_bucket_al);

	// thread-specific negative samplers
	vector<NegativeSampler> neg_samplers;
	for (int tid = 0; tid < num_thread; tid++)
		neg_samplers.push_back(NegativeSampler(ne, nr, rand() ^ tid));
		// neg_samplers.push_back(NegativeSampler(ne, nr, rand() ^ tid, hrt_bucket_tr));

	int N = hrts_tr.size();
	vector<int> pi = Range(N);

	clock_t start;
	double elapse_tr = 0;
	double elapse_ev = 0;
	double best_hit01 = 0;
	double best_mrr = 0;
	double best_hit10 = 0;

	omp_set_num_threads(num_thread);
	for (int epoch = 0; epoch < num_epoch; epoch++) {
		// evaluation
		if (epoch % eval_freq == 0) {
			start = omp_get_wtime();
			auto info_tr = evaluator_tr.evaluate(model, 2048);
			auto info_va = evaluator_va.evaluate(model, 2048);
			elapse_ev = omp_get_wtime() - start;

			// Save the model to disk
			double curr_mrr = (info_va["mrr_h"] + info_va["mrr_t"]) / 2.;
			double curr_hit10 = (info_va["hits10_h"] + info_va["hits10_t"]) / 2.;
			double curr_hit01 = (info_va["hits01_h"] + info_va["hits01_t"]) / 2.;
			if (curr_mrr > best_mrr)  {
				best_mrr = curr_mrr;
				if (!model_path_mrr.empty())
					model->Save(model_path_mrr);
			}

			if(curr_hit01 + curr_hit10 > best_hit01 + best_hit10){
				best_hit01 = curr_hit01;
				best_hit10 = curr_hit10;
				if (!model_path_mrr.empty())
					model->Save(model_path_hit);				
			}

			printf("\n");
			printf("            EV Elapse    %f\n", elapse_ev);
			printf("======================================\n");
			PrettyPrint("TR", info_tr);
			printf("\n");
			PrettyPrint("VA", info_va);
			printf("\n");
			printf("VA  BEST    %.2f\n", 100*best_mrr);
			printf("\n");
		}

		shuffle(pi.begin(), pi.end(), GLOBAL_GENERATOR);

		start = omp_get_wtime();

#pragma omp parallel for
			for (int i = 0; i < N; i++) {
				triplet hrt = hrts_tr[pi[i]];
				int h = get<0>(hrt);
				int r = get<1>(hrt);
				int t = get<2>(hrt);

				int tid = omp_get_thread_num();

			// positive example
				model->Train(h, r, t, 1.0);

			// negative examples
				for (int j = 0; j < neg_sample; j++) {
					int tt = neg_samplers[tid].RandomEntity();
					int hh = neg_samplers[tid].RandomEntity();					

					model->Train(h, r, tt, -1.0);
					model->Train(hh, r, t, -1.0);
					
					if (dataset != "YAGO37"){
						int rr = neg_samplers[tid].RandomRelation();
						model->Train(h, rr, t, -1.0);
					}
				}
			}
		elapse_tr = omp_get_wtime() - start;
		printf("Epoch %03d   TR Elapse    %f\n", epoch, elapse_tr);
	}
	return 0;
}
