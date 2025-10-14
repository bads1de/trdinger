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
            <p className="mt-3 mb-4">
              生成される戦略は、以下の4つの「遺伝子」の組み合わせで構成されます。GAはこれらの遺伝子を様々に組み替え、最適な組み合わせを探します。
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-secondary-900 p-4 rounded-lg border border-secondary-600">
                <h4 className="font-semibold text-white mb-2">テクニカル指標 (IndicatorGene)</h4>
                <p className="text-sm">SMA、RSI、MACDなど最大5つの指標とそのパラメータ（期間など）を組み合わせ。相場の方向性や勢いを判断する基本要素。</p>
              </div>
              <div className="bg-secondary-900 p-4 rounded-lg border border-secondary-600">
                <h4 className="font-semibold text-white mb-2">売買条件 (ConditionGene)</h4>
                <p className="text-sm">テクニカル指標を基にした具体的なエントリー・エグジットルール。「SMA5がSMA25を上回ったら買い」のような条件を設定。</p>
              </div>
              <div className="bg-secondary-900 p-4 rounded-lg border border-secondary-600">
                <h4 className="font-semibold text-white mb-2">利食い・損切り (TPSLGene)</h4>
                <p className="text-sm">利益確定（テイクプロフィット）と損失限定（ストップロス）のルール。固定幅やATRを基準にし、リスク管理の重要な要素。</p>
              </div>
              <div className="bg-secondary-900 p-4 rounded-lg border border-secondary-600">
                <h4 className="font-semibold text-white mb-2">ポジションサイジング (PositionSizingGene)</h4>
                <p className="text-sm">1回の取引における投資額（ロット数）を決定。口座資金に対する固定比率やリスク量に応じた動的な計算方法で資金管理。</p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-xl text-white mb-3 border-b-2 border-blue-500 pb-2">探索プロセス (遺伝的アルゴリズム)</h3>
            <p className="mt-3 mb-6">
              生物の進化を模倣した遺伝的アルゴリズム(GA)により、以下の3つのフェーズを経て、段階的に戦略を洗練させていきます。
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border-2 border-blue-500 p-4 bg-secondary-900 rounded-lg shadow-lg">
                <h4 className="font-semibold text-lg text-white mb-3">フェーズ1: 多様性探索</h4>
                <p className="text-sm">
                  <strong>テクニカル指標</strong>と<strong>売買条件</strong>の組み合わせを幅広く探索。可能性の芽を探す段階。
                </p>
              </div>
              <div className="border-2 border-green-500 p-4 bg-secondary-900 rounded-lg shadow-lg">
                <h4 className="font-semibold text-lg text-white mb-3">フェーズ2: 精度向上</h4>
                <p className="text-sm">
                  有望な戦略のパラメータを微調整。<strong>利食い・損切り</strong>を最適化し収益性を高める段階。
                </p>
              </div>
              <div className="border-2 border-yellow-500 p-4 bg-secondary-900 rounded-lg shadow-lg">
                <h4 className="font-semibold text-lg text-white mb-3">フェーズ3: ポジションサイジング</h4>
                <p className="text-sm">
                  最終的な<strong>ポジションサイジング</strong>を組み込み。実践的な戦略を完成させる段階。
                </p>
              </div>
            </div>
          </div>
        </div>
        <div>
          <h3 className="font-semibold text-xl text-white mb-3 mt-6 border-b-2 border-blue-500 pb-2">実装の特徴</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-secondary-900 p-4 rounded-lg border border-secondary-600">
              <h4 className="font-semibold text-white mb-2">柔軟な戦略表現</h4>
              <p className="text-sm">戦略遺伝子はテクニカル指標・売買条件・TP/SL・ポジションサイジング・リスク管理・メタデータまでを統合的に表現でき、複雑な戦略も自動生成可能。</p>
            </div>
            <div className="bg-secondary-900 p-4 rounded-lg border border-secondary-600">
              <h4 className="font-semibold text-white mb-2">進化アルゴリズムの最適化</h4>
              <p className="text-sm">DEAPライブラリを活用し、個体生成・交叉・突然変異・選択・評価までを効率的に実装。進化過程で多様な戦略を探索し、最適解を自動的に発見。</p>
            </div>
            <div className="bg-secondary-900 p-4 rounded-lg border border-secondary-600">
              <h4 className="font-semibold text-white mb-2">評価指標の多様性</h4>
              <p className="text-sm">総リターン・シャープレシオ・ドローダウン・勝率など複数の指標を組み合わせてフィットネスを計算。ユーザー要件に応じて重みや制約も柔軟に設定可能。</p>
            </div>
            <div className="bg-secondary-900 p-4 rounded-lg border border-secondary-600">
              <h4 className="font-semibold text-white mb-2">拡張性・再現性</h4>
              <p className="text-sm">クラス設計が分離されており、戦略表現やGAパラメータ、評価関数の拡張が容易。進化過程や最良戦略はDBに保存され、再現性・検証性も確保。</p>
            </div>
          </div>
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
