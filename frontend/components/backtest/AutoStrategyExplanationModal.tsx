import React from 'react';

interface AutoStrategyExplanationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const AutoStrategyExplanationModal: React.FC<AutoStrategyExplanationModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-secondary-950 rounded-lg p-8 max-w-3xl w-full border border-secondary-700 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-2xl font-bold mb-6 text-white text-center">オートストラテジーとは？</h2>
        <div className="space-y-8 text-secondary-300">
          <div>
            <h3 className="font-semibold text-xl text-white mb-3 border-b-2 border-blue-500 pb-2">目的</h3>
            <p className="mt-3">
              遺伝的アルゴリズム(GA)を用いて、特定の金融商品や時間軸に最適な取引戦略を自動で探索・生成します。人間の直感や先入観に頼らず、データに基づいた客観的な戦略を見つけ出すことを目指します。
            </p>
          </div>

          <div>
            <h3 className="font-semibold text-xl text-white mb-3 border-b-2 border-blue-500 pb-2">戦略の構成要素</h3>
            <p className="mt-3">
              生成される戦略は、以下の4つの「遺伝子」の組み合わせで構成されます。GAはこれらの遺伝子を様々に組み替え、最適な組み合わせを探します。
            </p>
            <ul className="list-disc list-inside space-y-3 mt-4 pl-4">
              <li>
                <span className="font-semibold text-white">テクニカル指標 (IndicatorGene):</span>
                <span> SMA、RSI、MACDなど最大5つの指標と、そのパラメータ（期間など）を組み合わせます。相場の方向性や勢いを判断するための基本要素です。</span>
              </li>
              <li>
                <span className="font-semibold text-white">売買条件 (ConditionGene):</span>
                <span> テクニカル指標を基にした、具体的なエントリー・エグジットのルールです。「SMA5がSMA25を上回ったら買い」のような条件を、買い（ロング）と売り（ショート）それぞれで設定します。</span>
              </li>
              <li>
                <span className="font-semibold text-white">利食い・損切り (TPSLGene):</span>
                <span> 利益確定（テイクプロフィット）と損失限定（ストップロス）のルールです。固定幅やATR（Average True Range）を基準にするなど、リスク管理の重要な要素です。</span>
              </li>
              <li>
                <span className="font-semibold text-white">ポジションサイジング (PositionSizingGene):</span>
                <span> 1回の取引における投資額（ロット数）を決定します。口座資金に対する固定比率や、リスク量に応じた動的な計算方法などがあり、資金管理を司ります。</span>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-xl text-white mb-3 border-b-2 border-blue-500 pb-2">探索プロセス (遺伝的アルゴリズム)</h3>
            <p className="mt-3">
              生物の進化を模倣した遺伝的アルゴリズム(GA)により、以下の3つのフェーズを経て、段階的に戦略を洗練させていきます。
            </p>
            <div className="mt-5 space-y-5">
              <div className="border-l-4 border-blue-500 pl-4 py-3 bg-secondary-900 rounded-r-lg shadow-md">
                <h4 className="font-semibold text-lg text-white">フェーズ1: 多様性探索 (Diversity Exploration)</h4>
                <p className="mt-2">
                  この段階では、主に<strong>テクニカル指標</strong>と<strong>売買条件</strong>の様々な組み合わせをランダムに生成し、有望な戦略の「原型」を幅広く探索します。多様なアプローチを試し、可能性の芽を探すことを目的とします。
                </p>
              </div>
              <div className="border-l-4 border-green-500 pl-4 py-3 bg-secondary-900 rounded-r-lg shadow-md">
                <h4 className="font-semibold text-lg text-white">フェーズ2: 精度向上 (Refinement)</h4>
                <p className="mt-2">
                  フェーズ1で見つかった有望な戦略に絞り込み、そのパラメータを微調整することでパフォーマンスの向上を追求します。特に、<strong>利食い・損切り</strong>のルールを最適化し、より現実的な収益性を高めることを目指します。
                </p>
              </div>
              <div className="border-l-4 border-yellow-500 pl-4 py-3 bg-secondary-900 rounded-r-lg shadow-md">
                <h4 className="font-semibold text-lg text-white">フェーズ3: ポジションサイジング (Position Sizing)</h4>
                <p className="mt-2">
                  フェーズ2で最適化された戦略に、最終的な仕上げとして<strong>ポジションサイジング</strong>のルールを組み込みます。これにより、リスクを管理し、長期的に安定した運用を目指せる、より実践的な戦略が完成します。
                </p>
              </div>
            </div>
          </div>
        </div>
        <div>
          <h3 className="font-semibold text-xl text-white mb-3 mt-4 border-b-2 border-blue-500 pb-2">実装の特徴</h3>
          <ul className="list-disc list-inside space-y-3 mt-4 pl-4">
            <li>
              <span className="font-semibold text-white">柔軟な戦略表現:</span>
              <span>
                戦略遺伝子はテクニカル指標・売買条件・TP/SL・ポジションサイジング・リスク管理・メタデータまでを統合的に表現でき、複雑な戦略も自動生成可能です。
              </span>
            </li>
            <li>
              <span className="font-semibold text-white">進化アルゴリズムの最適化:</span>
              <span>
                DEAPライブラリを活用し、個体生成・交叉・突然変異・選択・評価までを効率的に実装。進化過程で多様な戦略を探索し、最適解を自動的に発見します。
              </span>
            </li>
            <li>
              <span className="font-semibold text-white">評価指標の多様性:</span>
              <span>
                総リターン・シャープレシオ・ドローダウン・勝率など複数の指標を組み合わせてフィットネスを計算。ユーザー要件に応じて重みや制約も柔軟に設定できます。
              </span>
            </li>
            <li>
              <span className="font-semibold text-white">拡張性・再現性:</span>
              <span>
                クラス設計が分離されており、戦略表現やGAパラメータ、評価関数の拡張が容易。進化過程や最良戦略はDBに保存され、再現性・検証性も確保されています。
              </span>
            </li>
          </ul>
        </div>
        <div className="mt-8 text-center">
          <button onClick={onClose} className="btn-primary px-8 py-2 rounded-lg transition-transform transform hover:scale-105">
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
};

export default AutoStrategyExplanationModal;
