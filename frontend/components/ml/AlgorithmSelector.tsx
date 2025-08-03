import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
  Alert,
  CircularProgress,
  Badge,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Speed as SpeedIcon,
  Accuracy as AccuracyIcon,
  Memory as MemoryIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { useAlgorithms, Algorithm } from '../../hooks/useAlgorithms';

interface AlgorithmSelectorProps {
  selectedAlgorithms?: string[];
  onSelectionChange?: (algorithms: string[]) => void;
  maxSelection?: number;
  showRecommendations?: boolean;
  requirements?: {
    dataSize?: 'small' | 'medium' | 'large';
    needsProbability?: boolean;
    needsFeatureImportance?: boolean;
    needsSpeed?: boolean;
    needsAccuracy?: boolean;
    hasNoise?: boolean;
  };
}

const AlgorithmSelector: React.FC<AlgorithmSelectorProps> = ({
  selectedAlgorithms = [],
  onSelectionChange,
  maxSelection = 5,
  showRecommendations = true,
  requirements = {},
}) => {
  const {
    algorithms,
    algorithmsByType,
    probabilityAlgorithms,
    featureImportanceAlgorithms,
    statistics,
    searchAlgorithms,
    getRecommendedAlgorithms,
    isLoading,
    error,
    getTypeLabel,
  } = useAlgorithms();

  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterCapability, setFilterCapability] = useState<string>('all');

  // フィルタリングされたアルゴリズム
  const filteredAlgorithms = useMemo(() => {
    let filtered = searchAlgorithms(searchQuery);

    if (filterType !== 'all') {
      filtered = filtered.filter(algo => algo.type === filterType);
    }

    if (filterCapability !== 'all') {
      filtered = filtered.filter(algo => 
        algo.capabilities.includes(filterCapability)
      );
    }

    return filtered;
  }, [searchAlgorithms, searchQuery, filterType, filterCapability]);

  // 推奨アルゴリズム
  const recommendedAlgorithms = useMemo(() => {
    if (!showRecommendations) return [];
    return getRecommendedAlgorithms(requirements);
  }, [getRecommendedAlgorithms, requirements, showRecommendations]);

  // アルゴリズム選択処理
  const handleAlgorithmToggle = (algorithmName: string) => {
    if (!onSelectionChange) return;

    const isSelected = selectedAlgorithms.includes(algorithmName);
    let newSelection: string[];

    if (isSelected) {
      newSelection = selectedAlgorithms.filter(name => name !== algorithmName);
    } else {
      if (selectedAlgorithms.length >= maxSelection) {
        return; // 最大選択数に達している
      }
      newSelection = [...selectedAlgorithms, algorithmName];
    }

    onSelectionChange(newSelection);
  };

  // アルゴリズムカードコンポーネント
  const AlgorithmCard: React.FC<{ algorithm: Algorithm; isRecommended?: boolean }> = ({ 
    algorithm, 
    isRecommended = false 
  }) => {
    const isSelected = selectedAlgorithms.includes(algorithm.name);
    const canSelect = !isSelected && selectedAlgorithms.length < maxSelection;

    return (
      <Card
        sx={{
          cursor: onSelectionChange ? 'pointer' : 'default',
          border: isSelected ? 2 : 1,
          borderColor: isSelected ? 'primary.main' : 'divider',
          backgroundColor: isSelected ? 'primary.50' : 'background.paper',
          position: 'relative',
          '&:hover': onSelectionChange ? {
            borderColor: 'primary.main',
            boxShadow: 2,
          } : {},
        }}
        onClick={() => onSelectionChange && canSelect && handleAlgorithmToggle(algorithm.name)}
      >
        {isRecommended && (
          <Chip
            label="推奨"
            color="secondary"
            size="small"
            sx={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}
          />
        )}
        
        <CardContent>
          <Box display="flex" alignItems="center" mb={1}>
            {isSelected && <CheckCircleIcon color="primary" sx={{ mr: 1 }} />}
            <Typography variant="h6" component="h3">
              {algorithm.display_name}
            </Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" mb={2}>
            {algorithm.description}
          </Typography>

          <Box mb={2}>
            <Chip
              label={getTypeLabel(algorithm.type)}
              size="small"
              variant="outlined"
              sx={{ mr: 1, mb: 1 }}
            />
            {algorithm.has_probability_prediction && (
              <Chip
                label="確率予測"
                size="small"
                color="info"
                sx={{ mr: 1, mb: 1 }}
              />
            )}
            {algorithm.has_feature_importance && (
              <Chip
                label="特徴量重要度"
                size="small"
                color="success"
                sx={{ mr: 1, mb: 1 }}
              />
            )}
          </Box>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2">詳細情報</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box>
                <Typography variant="subtitle2" color="success.main" gutterBottom>
                  長所:
                </Typography>
                <List dense>
                  {algorithm.pros.map((pro, index) => (
                    <ListItem key={index} sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 20 }}>
                        <CheckCircleIcon fontSize="small" color="success" />
                      </ListItemIcon>
                      <ListItemText primary={pro} />
                    </ListItem>
                  ))}
                </List>

                <Typography variant="subtitle2" color="warning.main" gutterBottom mt={1}>
                  短所:
                </Typography>
                <List dense>
                  {algorithm.cons.map((con, index) => (
                    <ListItem key={index} sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 20 }}>
                        <WarningIcon fontSize="small" color="warning" />
                      </ListItemIcon>
                      <ListItemText primary={con} />
                    </ListItem>
                  ))}
                </List>

                <Typography variant="subtitle2" color="info.main" gutterBottom mt={1}>
                  適用場面:
                </Typography>
                <List dense>
                  {algorithm.best_for.map((use, index) => (
                    <ListItem key={index} sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 20 }}>
                        <InfoIcon fontSize="small" color="info" />
                      </ListItemIcon>
                      <ListItemText primary={use} />
                    </ListItem>
                  ))}
                </List>

                {algorithm.note && (
                  <Alert severity="warning" sx={{ mt: 1 }}>
                    {algorithm.note}
                  </Alert>
                )}
              </Box>
            </AccordionDetails>
          </Accordion>
        </CardContent>
      </Card>
    );
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* 統計情報 */}
      {statistics && (
        <Box mb={3}>
          <Typography variant="h6" gutterBottom>
            利用可能なアルゴリズム ({statistics.total}個)
          </Typography>
          <Box display="flex" gap={1} flexWrap="wrap">
            <Chip
              icon={<TrendingUpIcon />}
              label={`確率予測対応: ${statistics.probabilityCount}個`}
              variant="outlined"
              color="info"
            />
            <Chip
              icon={<AccuracyIcon />}
              label={`特徴量重要度: ${statistics.featureImportanceCount}個`}
              variant="outlined"
              color="success"
            />
          </Box>
        </Box>
      )}

      {/* 選択状況 */}
      {onSelectionChange && (
        <Box mb={3}>
          <Typography variant="subtitle1" gutterBottom>
            選択済み: {selectedAlgorithms.length} / {maxSelection}
          </Typography>
          <Box display="flex" gap={1} flexWrap="wrap">
            {selectedAlgorithms.map(name => (
              <Chip
                key={name}
                label={algorithms.find(a => a.name === name)?.display_name || name}
                onDelete={() => handleAlgorithmToggle(name)}
                color="primary"
              />
            ))}
          </Box>
        </Box>
      )}

      {/* フィルター */}
      <Box mb={3}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="検索"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="アルゴリズム名、説明で検索..."
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>タイプ</InputLabel>
              <Select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                label="タイプ"
              >
                <MenuItem value="all">すべて</MenuItem>
                {statistics?.byType.map(({ type, count }) => (
                  <MenuItem key={type} value={type}>
                    {type} ({count})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>機能</InputLabel>
              <Select
                value={filterCapability}
                onChange={(e) => setFilterCapability(e.target.value)}
                label="機能"
              >
                <MenuItem value="all">すべて</MenuItem>
                {statistics?.byCapability.map(({ capability, count }) => (
                  <MenuItem key={capability} value={capability}>
                    {capability} ({count})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Box>

      {/* 推奨アルゴリズム */}
      {showRecommendations && recommendedAlgorithms.length > 0 && (
        <Box mb={4}>
          <Typography variant="h6" gutterBottom>
            推奨アルゴリズム
          </Typography>
          <Grid container spacing={2}>
            {recommendedAlgorithms.slice(0, 3).map(algorithm => (
              <Grid item xs={12} md={4} key={algorithm.name}>
                <AlgorithmCard algorithm={algorithm} isRecommended />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* アルゴリズム一覧 */}
      <Typography variant="h6" gutterBottom>
        全アルゴリズム
      </Typography>
      <Grid container spacing={2}>
        {filteredAlgorithms.map(algorithm => (
          <Grid item xs={12} md={6} lg={4} key={algorithm.name}>
            <AlgorithmCard algorithm={algorithm} />
          </Grid>
        ))}
      </Grid>

      {filteredAlgorithms.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          条件に一致するアルゴリズムが見つかりませんでした。
        </Alert>
      )}
    </Box>
  );
};

export default AlgorithmSelector;
