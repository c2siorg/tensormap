/**
 * Algorithmic Fairness & Demographic Parity Audit Panel
 * Audits disparate impact ratio, demographic parity, and equal opportunity across subgroups.
 * @module
 */

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Scale, CheckCircle2, AlertOctagon, Users, Info, Search } from "lucide-react";
import axios from "@/shared/Axios";
import { BACKEND_MODEL_ANALYSIS } from "@/constants/Urls";
import { FairnessAuditData } from "@/types/analysis";
import logger from "@/shared/logger";

interface FairnessAuditPanelProps {
  jobId: string;
  features?: string[];
}

export default function FairnessAuditPanel({ jobId, features = [] }: FairnessAuditPanelProps) {
  const [sensitiveFeature, setSensitiveFeature] = useState<string>(features[0] || "");
  const [data, setData] = useState<FairnessAuditData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [hasAudited, setHasAudited] = useState<boolean>(false);

  const handleAudit = async (featureToAudit?: string) => {
    const targetFeature = (featureToAudit !== undefined ? featureToAudit : sensitiveFeature).trim();
    if (!targetFeature) {
      setError("Please specify a sensitive attribute name (e.g. sex, race, age_group, or a dataset feature).");
      return;
    }

    setIsLoading(true);
    setError(null);
    setHasAudited(true);

    try {
      const response = await axios.get(`${BACKEND_MODEL_ANALYSIS}/${jobId}/fairness`, {
        params: { sensitive_feature: targetFeature },
      });
      setData(response.data as FairnessAuditData);
    } catch (err: any) {
      logger.error("Failed to fetch fairness audit:", err);
      const detail = err.response?.data?.detail || err.message || "Failed to compute fairness audit";
      setError(detail);
      setData(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Scale className="h-5 w-5 text-indigo-600" />
            <div>
              <CardTitle className="text-base">Algorithmic Fairness & Demographic Parity Audit</CardTitle>
              <p className="text-xs text-gray-500 mt-0.5">
                Audits model predictions against the EEOC 80% (4/5ths) rule, disparate impact, and equal opportunity
              </p>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Sensitive Feature Selector */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
          <label htmlFor="sensitive-feature-input" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">
            Sensitive Demographic Attribute
          </label>
          <div className="flex flex-col sm:flex-row gap-2">
            <div className="relative flex-1">
              <Input
                id="sensitive-feature-input"
                type="text"
                value={sensitiveFeature}
                onChange={(e) => setSensitiveFeature(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleAudit();
                  }
                }}
                placeholder="Enter sensitive attribute (e.g. sex, race, age, category)"
                className="w-full bg-white text-sm"
              />
            </div>
            <Button
              onClick={() => handleAudit()}
              disabled={isLoading || !sensitiveFeature.trim()}
              className="bg-indigo-600 hover:bg-indigo-700 text-white flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Auditing...
                </>
              ) : (
                <>
                  <Search className="h-4 w-4" />
                  Audit Fairness
                </>
              )}
            </Button>
          </div>

          {/* Quick select feature chips */}
          {features.length > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-gray-500 mr-1">Available features:</span>
              {features.slice(0, 8).map((feat) => (
                <button
                  key={feat}
                  type="button"
                  onClick={() => {
                    setSensitiveFeature(feat);
                    handleAudit(feat);
                  }}
                  className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                    sensitiveFeature === feat
                      ? "bg-indigo-50 border-indigo-300 text-indigo-700 font-medium"
                      : "bg-white border-gray-200 text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  {feat}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Error message */}
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-800">
              <strong>Audit Error:</strong> {error}
            </p>
          </div>
        )}

        {/* Initial Prompt */}
        {!hasAudited && !data && !isLoading && (
          <div className="text-center py-6 border border-dashed rounded-lg text-gray-500">
            <Users className="h-8 w-8 mx-auto text-gray-400 mb-2" />
            <p className="text-sm">Select or specify a sensitive feature above and click &quot;Audit Fairness&quot;.</p>
            <p className="text-xs text-gray-400 mt-1">
              Evaluates whether selection rates and recall parity adhere to ethical AI guidelines.
            </p>
          </div>
        )}

        {/* Audit Results */}
        {data && (
          <div className="space-y-6">
            {/* Summary metrics header */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
              <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">80% Rule Compliance</p>
                <div className="mt-2">
                  {data.overall_summary.compliant_with_80_percent_rule ? (
                    <Badge className="bg-green-600 text-white hover:bg-green-700 flex items-center gap-1 text-xs">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      Compliant (Pass)
                    </Badge>
                  ) : (
                    <Badge className="bg-amber-600 text-white hover:bg-amber-700 flex items-center gap-1 text-xs">
                      <AlertOctagon className="h-3.5 w-3.5" />
                      Disparity Detected
                    </Badge>
                  )}
                </div>
              </div>

              <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Privileged Baseline Group</p>
                <p className="text-base font-semibold text-gray-900 mt-1 truncate" title={data.privileged_group}>
                  {data.privileged_group}
                </p>
              </div>

              <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Evaluated Groups</p>
                <p className="text-base font-semibold text-gray-900 mt-1">
                  {data.overall_summary.groups_evaluated}
                </p>
              </div>

              <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Total Test Samples</p>
                <p className="text-base font-semibold text-gray-900 mt-1">
                  {data.overall_summary.total_samples.toLocaleString()}
                </p>
              </div>
            </div>

            {/* Subgroup Disparities Table */}
            {Object.keys(data.disparities).length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-800 mb-2 flex items-center gap-1.5">
                  <span>Disparity Benchmarks (vs. Baseline: {data.privileged_group})</span>
                </h4>
                <div className="overflow-x-auto border rounded-md">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-gray-50/75">
                        <TableHead className="font-semibold text-gray-700">Subgroup</TableHead>
                        <TableHead className="font-semibold text-gray-700 text-right">
                          Disparate Impact Ratio (DIR)
                        </TableHead>
                        <TableHead className="font-semibold text-gray-700 text-right">
                          Demographic Parity Diff (DPD)
                        </TableHead>
                        <TableHead className="font-semibold text-gray-700 text-right">
                          Equal Opportunity Diff (EOD)
                        </TableHead>
                        <TableHead className="font-semibold text-gray-700 text-right">
                          Average Odds Diff (AOD)
                        </TableHead>
                        <TableHead className="font-semibold text-gray-700 text-center">80% Rule</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Object.values(data.disparities).map((disp) => (
                        <TableRow key={disp.group} className="hover:bg-gray-50/50">
                          <TableCell className="font-medium text-gray-900">{disp.group}</TableCell>
                          <TableCell className="text-right font-mono text-sm">
                            <span
                              className={
                                disp.passes_80_percent_rule
                                  ? "text-green-700 font-medium"
                                  : "text-amber-700 font-bold"
                              }
                            >
                              {disp.disparate_impact_ratio.toFixed(4)}
                            </span>
                          </TableCell>
                          <TableCell className="text-right font-mono text-xs text-gray-700">
                            {disp.demographic_parity_difference > 0 ? "+" : ""}
                            {disp.demographic_parity_difference.toFixed(4)}
                          </TableCell>
                          <TableCell className="text-right font-mono text-xs text-gray-700">
                            {disp.equal_opportunity_difference > 0 ? "+" : ""}
                            {disp.equal_opportunity_difference.toFixed(4)}
                          </TableCell>
                          <TableCell className="text-right font-mono text-xs text-gray-700">
                            {disp.average_odds_difference > 0 ? "+" : ""}
                            {disp.average_odds_difference.toFixed(4)}
                          </TableCell>
                          <TableCell className="text-center">
                            {disp.passes_80_percent_rule ? (
                              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                                Pass
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                                Flagged (&lt; 0.8)
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            )}

            {/* Subgroup Metrics Table */}
            <div>
              <h4 className="text-sm font-semibold text-gray-800 mb-2">
                Subgroup Classification Performance Breakdown
              </h4>
              <div className="overflow-x-auto border rounded-md">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-gray-50/75">
                      <TableHead className="font-semibold text-gray-700">Subgroup</TableHead>
                      <TableHead className="font-semibold text-gray-700 text-right">Samples</TableHead>
                      <TableHead className="font-semibold text-gray-700 text-right">Selection Rate</TableHead>
                      <TableHead className="font-semibold text-gray-700 text-right">Accuracy</TableHead>
                      <TableHead className="font-semibold text-gray-700 text-right">Precision</TableHead>
                      <TableHead className="font-semibold text-gray-700 text-right">Recall (TPR)</TableHead>
                      <TableHead className="font-semibold text-gray-700 text-right">FPR</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.values(data.group_metrics).map((gm) => (
                      <TableRow key={gm.group} className="hover:bg-gray-50/50">
                        <TableCell className="font-medium text-gray-900">
                          <div className="flex items-center gap-1.5">
                            <span>{gm.group}</span>
                            {gm.group === data.privileged_group && (
                              <Badge variant="secondary" className="text-[10px] px-1 py-0 h-4">
                                Baseline
                              </Badge>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="text-right text-xs text-gray-600">{gm.sample_count}</TableCell>
                        <TableCell className="text-right font-mono text-sm font-medium text-gray-900">
                          {(gm.selection_rate * 100).toFixed(1)}%
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs text-gray-700">
                          {(gm.accuracy * 100).toFixed(1)}%
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs text-gray-700">
                          {(gm.precision * 100).toFixed(1)}%
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs text-gray-700">
                          {(gm.recall * 100).toFixed(1)}%
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs text-gray-700">
                          {(gm.false_positive_rate * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>

            {/* Explanatory note */}
            <div className="flex items-start gap-2 p-3 bg-blue-50/60 border border-blue-200/60 rounded-md text-xs text-blue-800">
              <Info className="h-4 w-4 shrink-0 mt-0.5 text-blue-600" />
              <div>
                <p className="font-semibold">EEOC 80% (Four-Fifths) Rule Guideline</p>
                <p className="mt-0.5 text-blue-700">
                  A selection rate for any demographic subgroup that is less than 80% (0.80) of the rate for the
                  highest-selected group is flagged as having disparate adverse impact. Demographic Parity and Equal
                  Opportunity differences close to 0.0 indicate balanced outcomes across groups.
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
