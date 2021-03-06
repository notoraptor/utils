/** Script to run Windows commands with specific environment variables values.
 * Intend to reproduce linux syntax:
 * 		VAR1=VAL1 VAR2=VAL2 ... command
 * with:
 * 		env VAR1=VAL1 VAR2=VAL2 ... command
 * Syntax used in Windows command line:
 * 		set VAR1=VAL1 && set VAR2=VAL2 && ... && command
 * Some documentation (although finally not used):
 * 		https://superuser.com/questions/223104/setting-and-using-variable-within-same-command-line-in-windows-cmd-exe
 * **/

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
using namespace std;

bool check_var_def(const char* arg, string& varname, string& varval) {
	if (isalpha(*arg)) {
		const char* s;
		for (s = arg + 1; *s != '\0' && *s != '='; ++s);
		if (*s == '=') {
			const char* t;
			for (t = arg + 1; t != s && (isalnum(*t) || *t == '_'); ++t);
			if (t == s) {
				varname.assign(arg, s - arg);
				++s;
				if (*s == '"') {
					const char* u;
					for (u = s + 1; *u != '\0'; ++u);
					if (*(u - 1) == '"') {
						varval.assign(s + 1, u - s - 2);
						return true;
					}
				} else {
					if (*s == '\0')
						varval.clear();
					else
						varval.assign(s);
					return true;
				}
			}
		}
	}
	return false;
}

bool check_command_piece(const char* c) {
	const char* s;
	for (s = c; *s != '\0' && !isspace(*s); ++s);
	return *s == '\0';
}

void add_protected(vector<string>& collection, const char* s) {
	if (check_command_piece(s))
		collection.push_back(s);
	else {
		string cpp_s = "\"";
		cpp_s += s;
		cpp_s += '"';
		collection.push_back(cpp_s);
	}
}

int main(int n, char* args[]) {
	if (n == 1) {
		cout << "Usage: " << args[0] << "VAR1=VAL1 VAR2=VAL2 ... command args ..." << endl;
		return 1;
	}
	vector<string> varnames, varvals, command;
	bool in_vars = true;
	for (int i = 1; i < n; ++i) {
		if (in_vars) {
			string varname, varval;
			in_vars = check_var_def(args[i], varname, varval);
			if (in_vars) {
				varnames.push_back(varname);
				varvals.push_back(varval);
			} else
				add_protected(command, args[i]);
		} else
			add_protected(command, args[i]);
	}
	if (!command.empty()) {
		#ifdef DEBUG
		cout << "Envionment variables:" << endl;
		for (size_t i = 0; i < varnames.size(); ++i)
			cout << "\t\"" << varnames[i] << "=" << varvals[i] << "\"" << endl;
		cout << "Command:" << endl << '\t' << command[0];
		for (size_t i = 1; i < command.size(); ++i) {
			cout << ' ' << command[i];
		}
		cout << endl;
		#endif
		ostringstream final_command;
		if (!varnames.empty()) {
			final_command << "set \"" << varnames[0] << '=' << varvals[0] << '"';
			for (size_t i = 1; i < varnames.size(); ++i)
				final_command << " && set \"" << varnames[i] << '=' << varvals[i] << '"';
			final_command << " && ";
		}
		final_command << command[0];
		for (size_t i = 1; i < command.size(); ++i)
			final_command << ' ' << command[i];
		#ifdef DEBUG
		cout << final_command.str() << endl;
		#endif
		return system(final_command.str().c_str());
	}
	return 0;
}
