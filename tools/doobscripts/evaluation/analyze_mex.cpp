/*
 *  Filename: analyze_mex.cpp
 *  Project: functionAnalyze
 *  Created Date: Saturday March 30th 2019
 *  Author: microhe
 *  -----
 *  Last Modified:
 *  Modified By:
 *  -----
 *  Copyright (c) 2019 microhe
 */

#include "mex.hpp"
#include "mexAdapter.hpp"
#include <map>
#include <set>
using namespace std;
using matlab::mex::ArgumentList;
using namespace matlab::data;

class MexFunction : public matlab::mex::Function {
    void operator()(matlab::mex::ArgumentList outputs,
        matlab::mex::ArgumentList inputs)
    {
        checkArguments(outputs, inputs);
        int fh_size = inputs[0].getNumberOfElements() / inputs[0].getDimensions().size();
        int ifi_size = inputs[1].getNumberOfElements();
        // stream << fh_size << " " << ifi_size << "\n";
        // displayOnMATLAB(stream);
        matlab::data::TypedArray<int> function_history = std::move(inputs[0]);
        matlab::data::TypedArray<int> ignore_function_id = std::move(inputs[1]);

        set<int> ignore_function_id_set;
        for (int i = 0; i < ifi_size; i++) {
            ignore_function_id_set.insert(ignore_function_id[i]);
        }

        int depth = 1;
        int function_id = 0;
        int state_code = 0;
        vector<pair<int, int>> depth_fid_vector;
        map<pair<int, int>, int> depth_fid_map;
        for (int i = 0; i < fh_size; i++) {
            state_code = function_history[0][i];
            function_id = function_history[1][i];
            if (ignore_function_id_set.find(function_id) != ignore_function_id_set.end()) {
                continue;
            }

            if (state_code == 0) {
                auto iter = depth_fid_map.find(make_pair(depth, function_id));
                if (iter == depth_fid_map.end()) {
                    depth_fid_vector.push_back(make_pair(depth, function_id));
                    depth_fid_map.insert(make_pair(make_pair(depth, function_id), 1));
                } else {
                    iter->second += 1;
                }
                depth += 1;
            } else if (state_code == 1 && depth > 0) {
                depth -= 1;
            }
        }
        int depth_fid_len = depth_fid_vector.size();
        int* tmp = new int[depth_fid_len * 3];
        int j = 0;
        for (int i = 0; i < depth_fid_len; i++) {
            tmp[j++] = depth_fid_vector[i].first;
            tmp[j++] = depth_fid_vector[i].second;
            auto iter = depth_fid_map.find(depth_fid_vector[i]);
            tmp[j++] = iter->second;
        }
        outputs[0] = factory.createArray<int>({ 3, depth_fid_len }, tmp, tmp + j);
        delete tmp;
    }
    void checkArguments(matlab::mex::ArgumentList outputs,
        matlab::mex::ArgumentList inputs)
    {
        // std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        // matlab::data::ArrayFactory factory;

        if (inputs.size() != 2) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Two inputs required.") }));
        }

        if (inputs[0].getType() != matlab::data::ArrayType::INT32 || inputs[0].getType() == matlab::data::ArrayType::COMPLEX_INT32) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input penalties must be type int.") }));
        }

        if (inputs[0].getDimensions().size() != 2) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input must be m-by-n dimension") }));
        }

        if (inputs[1].getType() != matlab::data::ArrayType::INT32 || inputs[1].getType() == matlab::data::ArrayType::COMPLEX_INT32) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input matrix must be type int") }));
        }
    }
    void displayOnMATLAB(std::ostringstream& stream)
    {
        // Pass stream content to MATLAB fprintf function
        matlabPtr->feval(u"fprintf", 0,
            std::vector<Array>({ factory.createScalar(stream.str()) }));
        // Clear stream buffer
        stream.str("");
    }

private:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    // Factory to create MATLAB data arrays
    matlab::data::ArrayFactory factory;

    // Create an output stream
    std::ostringstream stream;
};